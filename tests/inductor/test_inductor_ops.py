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
    cached_xavier,
    make_param_dict,
    unique_randn_along_dim,
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
    # See utils_inductor.py for more details.
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
                    ((7, 12, 32, 64),) * 2,
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
        ("test_addmm", "test_addmm_cpu"): {
            "param_sets": make_param_dict(
                [
                    ((1152,), (10, 1152), (1152, 1152)),
                ],
            ),
        },
        ("test_mm", "test_mm_relaxed"): {
            "ops_dict": {
                "mm": torch.mm,
                # "einsum": lambda a, b: torch.einsum('mk, kn -> mn', a, b),  # bmm not supported yet
            },
            "param_sets": make_param_dict(
                [
                    ((67, 256), (256, 128)),
                    # Padding
                    ((55, 2), (2, 99)),
                    ((67, 67), (67, 67)),
                    ((67, 255), (255, 128)),
                ]
            ),
        },
        ("test_bmm", "test_mm_relaxed"): {
            "ops_dict": {"bmm": torch.bmm},
            "param_sets": make_param_dict(
                [
                    ((3, 1, 256), (3, 256, 128)),
                    ((3, 17, 256), (3, 256, 128)),
                    ((2, 256, 1), (2, 1, 128)),
                    # Padding
                    ((2, 55, 2), (2, 2, 99)),
                    ((2, 99, 65), (2, 65, 55)),
                ]
            ),
        },
        ("test_matmul", "test_binary_op_cpu"): {
            "ops_dict": {
                "matmul": torch.matmul,
            },
            "param_sets": make_param_dict(
                [
                    ((512, 256), (256, 128)),
                    ((3, 1, 256), (3, 256, 128)),
                    ((3, 17, 256), (3, 256, 128)),
                    # Modify the second dimension from 17 to 18 to avoid the issue of a prime
                    # tensor shape until https://github.com/torch-spyre/torch-spyre/issues/399
                    # is resolved.
                    ((3, 18, 128, 256), (3, 18, 256, 128)),
                    ((2, 64, 128), (128, 16384)),
                    ((99, 1), (1, 55)),
                    ((2, 99, 1), (2, 1, 55)),
                    ((2, 99, 1), (1, 55)),
                    ((2, 3, 99, 1), (2, 3, 1, 55)),
                    # Test padding for mm/bmm.
                    ((55, 2), (2, 99)),
                    ((99, 65), (65, 55)),
                    ((2, 55, 2), (2, 2, 99)),
                    ((2, 99, 65), (2, 65, 55)),
                    ((2, 3, 55, 2), (2, 3, 2, 99)),
                    ((2, 3, 99, 65), (2, 3, 65, 55)),
                ]
            ),
        },
        ("test_large_matmul", "test_mm_relaxed"): {
            "ops_dict": {"matmul": torch.matmul},
            "param_sets": {
                "2d_M2048_K2048_N65536": (
                    cached_randn((2048, 2048)),
                    cached_xavier((2048, 65536)),
                ),
                "3d_M3_K11_N2880": (
                    cached_randn((3, 11, 2880)),
                    cached_xavier((3, 2880, 2880)),
                ),
                "3d2d_M3_K11_N2880": (
                    cached_randn((3, 11, 2880)),
                    cached_xavier((2880, 2880)),
                ),
                "4d_B2_H2_M2048_K2048_N65536": (
                    cached_randn((2, 2, 2048, 2048)),
                    cached_xavier((2, 2, 2048, 65536)),
                ),
            },
        },
        ("test_sdsc_padding_sum_keepdim1", "test_reduce_keepdim1_cpu"): {
            "ops_dict": {"sum": torch.sum},
            "param_sets": {
                "2d_0": (0, cached_randn((63, 129))),
                "2d_1": (1, cached_randn((63, 129))),
                # Skip until https://github.com/torch-spyre/torch-spyre/issues/521 is implemented
                # "2d_01": ((0, 1), cached_randn((63, 129))),
                "3d_0": (0, cached_randn((3, 7, 9))),
                "3d_1": (1, cached_randn((3, 7, 9))),
                "3d_2": (2, cached_randn((3, 7, 9))),
                "3d_01": ((0, 1), cached_randn((3, 7, 9))),
                "3d_12": ((1, 2), cached_randn((3, 7, 9))),
                # Skip until https://github.com/torch-spyre/torch-spyre/issues/521 is implemented
                # "3d_012": ((0, 1, 2), cached_randn((3, 7, 9))),
                "4d_0": (0, cached_randn((3, 7, 9, 32))),
                "4d_1": (1, cached_randn((3, 7, 9, 32))),
                "4d_2": (2, cached_randn((3, 7, 9, 32))),
                "4d_3": (3, cached_randn((3, 7, 9, 32))),
            },
        },
        ("test_sdsc_padding_amin_keepdim1", "test_reduce_keepdim1_cpu_no_eager"): {
            "ops_dict": {"amin": torch.amin},
            "param_sets": {
                "dim_0": (0, unique_randn_along_dim((3, 7), dim=0)),
                "dim_1": (1, unique_randn_along_dim((3, 7), dim=1)),
                #  Disabled because torch-sendnn fails
                # "dim_01": ([0, 1], torch.ones((3, 7), dtype=torch.float16)),
            },
        },
        ("test_amax_keepdim1", "test_reduce_keepdim1_cpu"): {
            "ops_dict": {"amax": torch.amax},
            "param_sets": {
                # 1D tensor
                "1d_dim_0": (0, cached_randn((10,))),
                "1d_dim_none": (None, cached_randn((10,))),
                # 2D tensor
                "2d_dim_0": (0, cached_randn((67, 256))),
                "2d_dim_1": (1, cached_randn((67, 256))),
                "2d_dim_none": (None, cached_randn((67, 256))),
                # 3D tensor
                "3d_dim_0": (0, cached_randn((3, 7, 9))),
                "3d_dim_1": (1, cached_randn((3, 7, 9))),
                "3d_dim_2": (2, cached_randn((3, 7, 9))),
                "3d_dim_none": (None, cached_randn((3, 7, 9))),
                "3d_dim_01": ((0, 1), cached_randn((3, 7, 9))),
                "3d_dim_12": ((1, 2), cached_randn((3, 7, 9))),
                "3d_dim_012": ((0, 1, 2), cached_randn((3, 7, 9))),
                "3d_dim_unsorted": ((2, 0), cached_randn((3, 7, 9))),
                # Negative dims
                "3d_dim_neg1": (-1, cached_randn((3, 7, 9))),
                "3d_dim_neg12": ((-1, -2), cached_randn((3, 7, 9))),
                # 0D / scalar tensor
                # "scalar_tensor": (None, torch.tensor(5.0, dtype=torch.float16)), #TODO
            },
        },
        ("test_amax_keepdim0", "test_reduce_keepdim0_cpu"): {
            "ops_dict": {"amax": torch.amax},
            "param_sets": {
                # 1D tensor
                "1d_dim_0": (0, cached_randn((10,))),
                "1d_dim_none": (None, cached_randn((10,))),
                # 2D tensor
                "2d_dim_0": (0, cached_randn((67, 256))),
                "2d_dim_1": (1, cached_randn((67, 256))),
                "2d_dim_none": (None, cached_randn((67, 256))),
                # 3D tensor
                "3d_dim_0": (0, cached_randn((3, 7, 9))),
                "3d_dim_1": (1, cached_randn((3, 7, 9))),
                "3d_dim_2": (2, cached_randn((3, 7, 9))),
                "3d_dim_none": (None, cached_randn((3, 7, 9))),
                "3d_dim_01": ((0, 1), cached_randn((3, 7, 9))),
                "3d_dim_12": ((1, 2), cached_randn((3, 7, 9))),
                "3d_dim_012": ((0, 1, 2), cached_randn((3, 7, 9))),
                "3d_dim_unsorted": ((2, 0), cached_randn((3, 7, 9))),
                # Negative dims
                "3d_dim_neg1": (-1, cached_randn((3, 7, 9))),
                "3d_dim_neg12": ((-1, -2), cached_randn((3, 7, 9))),
                # 0D / scalar tensor:
                "scalar_tensor": (None, torch.tensor(5.0, dtype=torch.float16)),
            },
        },
        ("test_max_sub_broadcast", "test_max_sub_broadcast"): {
            "param_sets": {
                "2d_dim_0": (0, cached_randn((128, 256))),
                "2d_dim_1": (1, cached_randn((128, 256))),
                "4d_dim_0": (0, cached_randn((12, 8, 25, 64))),
                "4d_dim_1": (1, cached_randn((12, 8, 25, 64))),
                "4d_dim_2": (2, cached_randn((12, 8, 25, 64))),
                "4d_dim_3": (3, cached_randn((12, 8, 25, 64))),
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
                    ((67, 71, 256),),
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
                    ((67, 71, 256),),
                ]
            ),
        },
        # TODO(imaihal): Need to fix scalar tensor shape mismatch during Spyre-to-CPU transfer.
        # Spyre represents scalars as 1D tensors [1], while CPU uses 0D tensors.
        # This inconsistency causes shape mismatches when moving tensors from Spyre to CPU.
        # See: https://github.com/torch-spyre/torch-spyre/issues/1172
        # ("test_max_default", "test_reduce_cpu"): {
        #        "ops_dict": {
        #            "max": torch.max,
        #        },
        #        "param_sets": {
        #            "1d_float16": (unique_randn_along_dim((64,), dtype=torch.float16),),
        #            "2d_float16": (unique_randn_along_dim((8, 64), dtype=torch.float16),),
        #            "3d_float16": (
        #                unique_randn_along_dim((2, 4, 64), dtype=torch.float16),
        #            ),
        #            "1d_int64": (unique_randn_along_dim((64,), dtype=torch.int64),),
        #            "2d_int64": (unique_randn_along_dim((67, 256), dtype=torch.int64),),
        #        },
        #    },
        # Compare with cpu for now to avoid hitting eager mode coverage issue
        ("test_max_keepdim0", "test_reduce_keepdim0_cpu_no_eager"): {
            "ops_dict": {
                "max": torch.max,
            },
            "param_sets": {
                "2d_dim_0": (0, unique_randn_along_dim((67, 256), dim=0)),
                "2d_dim_1": (
                    1,
                    unique_randn_along_dim((67, 256), dim=1),
                ),  #  sparse tensor output
                "3d_dim_0": (
                    0,
                    unique_randn_along_dim((67, 71, 256), dim=0),
                ),  # layout needs repermutation
                "3d_dim_1": (1, unique_randn_along_dim((67, 71, 256), dim=1)),
                "3d_dim_2": (
                    2,
                    unique_randn_along_dim((67, 71, 256), dim=2),
                ),  # sparse tensor output
                "4d_dim_0": (0, unique_randn_along_dim((6, 17, 7, 64), dim=0)),
                "4d_dim_1": (1, unique_randn_along_dim((6, 17, 7, 64), dim=1)),
                "4d_dim_2": (2, unique_randn_along_dim((6, 17, 7, 64), dim=2)),
                "4d_dim_3": (
                    3,
                    unique_randn_along_dim((6, 17, 7, 64), dim=3),
                ),  # sparse tensor output
                "4d_dim_gpt0": (
                    -1,
                    unique_randn_along_dim((1, 64, 1, 129), dim=-1),
                ),  # gpt_oss
                "4d_dim_gpt1": (
                    -1,
                    unique_randn_along_dim((1, 64, 11, 129), dim=-1),
                ),  # gpt_oss
                "2d_dim_0_int64": (
                    0,
                    unique_randn_along_dim(
                        (67, 256), dim=0, min_val=0, max_val=100, dtype=torch.int64
                    ),
                ),
                "2d_dim_1_int64": (
                    1,
                    unique_randn_along_dim(
                        (67, 256), dim=1, min_val=0, max_val=100, dtype=torch.int64
                    ),
                ),
            },
        },
        ("test_max_keepdim1", "test_reduce_keepdim1_cpu_no_eager"): {
            "ops_dict": {
                "max": torch.max,
            },
            "param_sets": {
                "2d_dim_0": (0, unique_randn_along_dim((67, 256), dim=0)),
                "2d_dim_1": (
                    1,
                    unique_randn_along_dim((67, 256), dim=1),
                ),  # sparse tensor output
                "3d_dim_0": (0, unique_randn_along_dim((67, 71, 256), dim=0)),
                "3d_dim_1": (1, unique_randn_along_dim((67, 71, 256), dim=1)),
                "3d_dim_2": (
                    2,
                    unique_randn_along_dim((67, 71, 256), dim=2),
                ),  # sparse tensor output
                "4d_dim_0": (0, unique_randn_along_dim((6, 7, 12, 256), dim=0)),
                "4d_dim_1": (1, unique_randn_along_dim((6, 7, 12, 256), dim=1)),
                "4d_dim_2": (2, unique_randn_along_dim((6, 7, 12, 256), dim=2)),
                "4d_dim_3": (3, unique_randn_along_dim((6, 7, 12, 256), dim=3)),
                "2d_dim_0_int64": (
                    0,
                    unique_randn_along_dim(
                        (67, 256), dim=0, min_val=0, max_val=100, dtype=torch.int64
                    ),
                ),
                "2d_dim_1_int64": (
                    1,
                    unique_randn_along_dim(
                        (67, 256), dim=1, min_val=0, max_val=100, dtype=torch.int64
                    ),
                ),
            },
        },
        ("test_sum_keepdim0", "test_reduce_keepdim0_cpu"): {
            "ops_dict": {
                "sum": torch.sum,
            },
            "param_sets": {
                "2d_dim_0": (0, cached_randn((67, 256))),
                "2d_dim_1": (1, cached_randn((67, 256))),  # sparse tensor output
                # "2d_dim_01": ([0, 1], cached_randn((67, 256))), # spyre scalar represented as 1d instead of 0d
                # "3d_dim_0": (0, cached_randn((67, 71, 256), scale=0.01)), # layout needs repermutation
                "3d_dim_1": (1, cached_randn((67, 71, 256), scale=0.01)),
                "3d_dim_2": (
                    2,
                    cached_randn((67, 71, 256), scale=0.01),
                ),  # sparse tensor output
                # Skip until https://github.com/torch-spyre/torch-spyre/issues/521 is implemented
                # "3d_dim_01": ([0, 1], cached_randn((67, 71, 256), scale=0.01)),
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
                # Skip until https://github.com/torch-spyre/torch-spyre/issues/521 is implemented
                # "2d_dim_01": ([0, 1], cached_randn((67, 256))),
                "3d_dim_0": (0, cached_randn((3, 5, 256), scale=0.1)),
                "3d_dim_1": (1, cached_randn((67, 71, 256), scale=0.1)),
                "3d_dim_2": (
                    2,
                    cached_randn((67, 71, 256), scale=0.1),
                ),  # sparse tensor output
                # Skip until https://github.com/torch-spyre/torch-spyre/issues/521 is implemented
                # "3d_dim_01": ([0, 1], cached_randn((67, 71, 256), scale=0.1)),
                # "3d_dim_012": ([0, 1, 2], cached_randn((67, 71, 256), scale=0.1)),
                "4d_dim_0": (0, cached_randn((6, 7, 12, 256), scale=0.1)),
                "4d_dim_1": (1, cached_randn((6, 7, 12, 256), scale=0.1)),
                "4d_dim_2": (2, cached_randn((6, 7, 12, 256), scale=0.1)),
                "4d_dim_3": (3, cached_randn((6, 7, 12, 256), scale=0.1)),
            },
        },
        ("test_t_1d", "test_t_1d_cpu"): {
            "param_sets": make_param_dict(
                [
                    ((3,),),
                ]
            ),
        },
        ("test_t_1d_contiguous", "test_t_1d_contiguous_cpu"): {
            "param_sets": make_param_dict(
                [
                    ((3,),),
                ]
            ),
        },
        ("test_t_2d", "test_t_2d_cpu"): {
            "param_sets": make_param_dict(
                [
                    ((1088, 320),),
                    ((320, 320),),
                ]
            ),
        },
        ("test_t_2d_contiguous", "test_t_2d_contiguous_cpu"): {
            "param_sets": make_param_dict(
                [
                    ((1088, 320),),
                    ((320, 320),),
                    ((49280, 4096),),
                    ((4096, 49280),),
                ]
            ),
        },
        ("test_transpose_2d", "test_transpose_2d_cpu"): {
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
                "dim_0_1": (
                    0,
                    1,
                    cached_randn((128, 64, 128), abs=True),
                ),
            }
        },
        ("test_transpose_2d_contiguous", "test_transpose_2d_contiguous_cpu"): {
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
                "dim_0_1": (
                    0,
                    1,
                    cached_randn((128, 64, 128), abs=True),
                ),
            }
        },
        ("test_transpose_3d", "test_transpose_3d_cpu"): {
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
                "dim_0_1": (
                    0,
                    1,
                    cached_randn((128, 64, 128), abs=True),
                ),
            }
        },
        ("test_transpose_3d_contiguous", "test_transpose_3d_contiguous_cpu"): {
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
                "dim_0_1": (
                    0,
                    1,
                    cached_randn((128, 64, 128), abs=True),
                ),
            }
        },
        ("test_transpose_4d", "test_transpose_4d_cpu"): {
            "param_sets": {
                "dim_0_3": (
                    0,
                    3,
                    cached_randn((256, 3, 17, 64), abs=True),
                ),
                "dim_2_3": (
                    2,
                    3,
                    cached_randn((3, 17, 128, 256), abs=True),
                ),
                "dim_1_3": (
                    1,
                    3,
                    cached_randn((3, 256, 17, 64), abs=True),
                ),
                "dim_1_2": (
                    1,
                    3,
                    cached_randn((3, 256, 64, 64), abs=True),
                ),
                "dim_0_1": (
                    0,
                    1,
                    cached_randn((64, 25, 7, 64), abs=True),
                ),
            }
        },
        ("test_transpose_4d_contiguous", "test_transpose_4d_contiguous_cpu"): {
            "param_sets": {
                "dim_0_3": (
                    0,
                    3,
                    cached_randn((256, 3, 17, 64), abs=True),
                ),
                "dim_2_3": (
                    2,
                    3,
                    cached_randn((3, 17, 128, 256), abs=True),
                ),
                "dim_1_3": (
                    1,
                    3,
                    cached_randn((3, 256, 17, 64), abs=True),
                ),
                "dim_1_2": (
                    1,
                    3,
                    cached_randn((3, 256, 64, 64), abs=True),
                ),
                "dim_0_1": (
                    0,
                    1,
                    cached_randn((64, 25, 7, 64), abs=True),
                ),
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
                "sigmoid": torch.sigmoid,
                "mish": torch.nn.functional.mish,
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
                "fp16_1d": (cached_randn((2,), dtype=torch.float16),),
                "fp16_2d": (cached_randn((256, 100), dtype=torch.float16),),
                "fp16_3d": (cached_randn((8, 16, 256), dtype=torch.float16),),
                "fp16_4d": (cached_randn((8, 2, 16, 250), dtype=torch.float16),),
                "fp32_1d": (cached_randn((128,), dtype=torch.float32),),
                "fp32_2d": (cached_randn((256, 128), dtype=torch.float32),),
                "fp32_3d": (cached_randn((8, 16, 26), dtype=torch.float32),),
                "bool_1d": (torch.rand((128,)) > 0.5,),
                "bool_2d": (torch.rand((256, 128)) > 0.5,),
                "bool_3d": (torch.rand((8, 16, 256)) > 0.5,),
            },
        },
        (
            "test_permute",
            "test_permute",
        ): {
            "param_sets": {
                "2d_1_0": ((2, 3), (1, 0)),
                "4d_0_2_1_3": ((2, 3, 16, 64), (0, 2, 1, 3)),
                "3d_0_2_1": ((2, 1024, 844), (0, 2, 1)),
                "4d_0_3_1_2": ((2, 2, 256, 48), (0, 3, 1, 2)),
                "4d_0_m2_m1_1": ((2, 48, 2, 256), (0, -2, -1, 1)),
                "5d_0_2_3_4_1": ((2, 48, 2, 256, 265), (0, 2, 3, 4, 1)),
            },
        },
        (
            "test_cat",
            "test_cat_cpu",
        ): {
            "param_sets": {
                "1d_dim0": (
                    0,
                    cached_randn((64,), dtype=torch.float16),
                    cached_randn((128,), dtype=torch.float16),
                ),
                "1d_dim0_three_tensors": (
                    0,
                    cached_randn((64,), dtype=torch.float16),
                    cached_randn((128,), dtype=torch.float16),
                    cached_randn((192,), dtype=torch.float16),
                ),
                "2d_dim0_diff_size": (
                    0,
                    cached_randn((64, 128), dtype=torch.float16),
                    cached_randn((128, 128), dtype=torch.float16),
                ),
                "2d_dim0_three_tensors": (
                    0,
                    cached_randn((64, 64), dtype=torch.float16),
                    cached_randn((128, 64), dtype=torch.float16),
                    cached_randn((192, 64), dtype=torch.float16),
                ),
                "2d_dim1_diff_size": (
                    1,
                    cached_randn((128, 64), dtype=torch.float16),
                    cached_randn((128, 128), dtype=torch.float16),
                ),
                "3d_dim0": (
                    0,
                    cached_randn((2, 32, 64), dtype=torch.float16),
                    cached_randn((3, 32, 64), dtype=torch.float16),
                ),
                "3d_dim1": (
                    1,
                    cached_randn((2, 32, 64), dtype=torch.float16),
                    cached_randn((2, 16, 64), dtype=torch.float16),
                ),
                "3d_dim2": (
                    2,
                    cached_randn((2, 32, 64), dtype=torch.float16),
                    cached_randn((2, 32, 128), dtype=torch.float16),
                ),
                "3d_dim1_size1": (
                    1,
                    cached_randn((8, 64, 128), dtype=torch.float16),
                    cached_randn((8, 1, 128), dtype=torch.float16),
                ),
                "4d_dim0": (
                    0,
                    cached_randn((2, 4, 8, 64), dtype=torch.float16),
                    cached_randn((3, 4, 8, 64), dtype=torch.float16),
                ),
                "4d_dim1": (
                    1,
                    cached_randn((2, 4, 8, 64), dtype=torch.float16),
                    cached_randn((2, 6, 8, 64), dtype=torch.float16),
                ),
                "4d_dim2": (
                    2,
                    cached_randn((2, 4, 8, 64), dtype=torch.float16),
                    cached_randn((2, 4, 12, 64), dtype=torch.float16),
                ),
                "4d_dim3": (
                    3,
                    cached_randn((2, 4, 8, 64), dtype=torch.float16),
                    cached_randn((2, 4, 8, 128), dtype=torch.float16),
                ),
                "4d_dim3_fp32": (
                    3,
                    cached_randn((2, 4, 3, 64), dtype=torch.float32),
                    cached_randn((2, 4, 3, 32), dtype=torch.float32),
                ),
            },
        },
        (
            "test_pad",
            "test_pad_cpu",
        ): {
            "param_sets": {
                "2d_last_dim_right": (
                    cached_randn((3, 64), dtype=torch.float16),
                    (0, 64),
                ),
                "2d_both_dims": (
                    cached_randn((3, 64), dtype=torch.float16),
                    (0, 64, 0, 2),
                ),
                "3d_last_dim_right": (
                    cached_randn((2, 3, 64), dtype=torch.float16),
                    (0, 64),
                ),
                "3d_dim1_right": (
                    cached_randn((2, 3, 64), dtype=torch.float16),
                    (0, 0, 0, 2),
                ),
                "2d_last_dim_left_stick_aligned": (
                    cached_randn((3, 64), dtype=torch.float16),
                    (64, 0),
                ),
                "2d_last_dim_left_two_sticks": (
                    cached_randn((3, 64), dtype=torch.float16),
                    (128, 0),
                ),
                "2d_last_dim_left_and_right_stick_aligned": (
                    cached_randn((3, 64), dtype=torch.float16),
                    (64, 64),
                ),
                "2d_dim0_left": (
                    cached_randn((3, 64), dtype=torch.float16),
                    (0, 0, 2, 0),
                ),
                "2d_dim0_left_only": (
                    cached_randn((3, 64), dtype=torch.float16),
                    (0, 0, 1, 0),
                ),
                "3d_dim0_left": (
                    cached_randn((2, 3, 64), dtype=torch.float16),
                    (0, 0, 0, 0, 2, 0),
                ),
                "3d_dim1_left": (
                    cached_randn((2, 3, 64), dtype=torch.float16),
                    (0, 0, 1, 0),
                ),
                "4d_dim0_left": (
                    cached_randn((2, 3, 4, 64), dtype=torch.float16),
                    (0, 0, 0, 0, 0, 0, 1, 0),
                ),
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
            "test_empty_like",
            "test_empty_like_cpu",
        ): {
            "param_sets": {
                "1d_fp16": (cached_randn((64,), dtype=torch.float16),),
                "2d_fp16": (cached_randn((4, 8), dtype=torch.float16),),
                "2d_fp32": (cached_randn((4, 8), dtype=torch.float32),),
                "3d_fp16": (cached_randn((2, 4, 8), dtype=torch.float16),),
            },
        },
        (
            "test_empty_like_dtype_override",
            "test_empty_like_dtype_override_cpu",
        ): {
            "param_sets": {
                "fp16_to_fp32": (cached_randn((4, 8), dtype=torch.float16),),
                "fp32_to_fp16": (cached_randn((4, 8), dtype=torch.float32),),
            },
        },
        (
            "test_empty_like_memory_format",
            "test_empty_like_memory_format_cpu",
        ): {
            "param_sets": {
                "transposed_2d": (cached_randn((4, 8), dtype=torch.float16),),
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
            "test_ones",
            "test_ones_cpu",
        ): {
            "param_sets": {
                "1d": ((64,),),
                "2d_square": ((64, 64),),
                "2d": ((64, 128),),
                "3d": ((4, 3, 64),),
                "2d_padded": ((3, 50),),
            },
        },
        (
            "test_numel",
            "test_numel_cpu",
        ): {
            "param_sets": {
                "size_1": (cached_randn((64, 128)),),
            },
        },
        (
            "test_full",
            "test_full_cpu",
        ): {
            "param_sets": {
                "value_1": (([64, 128]), -65472.0),
                "value_2": (([64, 128]), -65504.0),
                "tuple": (((64, 64)), 1024.0),
                "size": (torch.Size([64, 128]), 1024.0),
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
                    cached_randn((1, 64, 11, 512)),
                    {
                        "p": 0.0,
                        "training": False,
                        "inplace": False,
                    },
                ),
            },
        },
        ("test_softmax", "test_dim_op_cpu_eager"): {
            "ops_dict": {
                "softmax": lambda dim, x: torch.softmax(x, dim=dim),
            },
            "param_sets": {
                "2d_dim0": (0, cached_randn((512, 1024), dtype=torch.float16)),
                "2d_dim1": (1, cached_randn((512, 1024), dtype=torch.float16)),
                "3d_dim0": (0, cached_randn((256, 64, 128), dtype=torch.float16)),
                "3d_dim1": (1, cached_randn((256, 64, 128), dtype=torch.float16)),
                "3d_dim2": (2, cached_randn((256, 64, 128), dtype=torch.float16)),
                "4d_dim0": (0, cached_randn((6, 17, 32, 64), dtype=torch.float16)),
                "4d_dim1": (1, cached_randn((6, 17, 32, 64), dtype=torch.float16)),
                "4d_dim2": (2, cached_randn((6, 17, 32, 64), dtype=torch.float16)),
                "4d_dim3": (3, cached_randn((6, 17, 32, 64), dtype=torch.float16)),
            },
        },
        (
            "test_size_one",
            "test_unary_op_cpu",
        ): {
            "ops_dict": {
                "exp": torch.exp,
            },
            "param_sets": {
                "1d0": {cached_randn((1,), dtype=torch.float16)},
                "2d0": {cached_randn((1, 3), dtype=torch.float16)},
                "2d1": {cached_randn((2, 1), dtype=torch.float16)},
                "3d0": {cached_randn((1, 3, 4), dtype=torch.float16)},
                "3d1": {cached_randn((2, 1, 4), dtype=torch.float16)},
                "3d2": {cached_randn((2, 3, 1), dtype=torch.float16)},
                "3d01": {cached_randn((1, 1, 4), dtype=torch.float16)},
                "3d02": {cached_randn((2, 3, 1), dtype=torch.float16)},
                "3d12": {cached_randn((1, 1, 4), dtype=torch.float16)},
                "4d0": {cached_randn((1, 3, 4, 5), dtype=torch.float16)},
                "4d1": {cached_randn((2, 1, 4, 5), dtype=torch.float16)},
                "4d2": {cached_randn((2, 3, 1, 5), dtype=torch.float16)},
                "4d3": {cached_randn((2, 3, 4, 1), dtype=torch.float16)},
                "4d01": {cached_randn((1, 1, 4, 5), dtype=torch.float16)},
                "4d02": {cached_randn((1, 3, 1, 5), dtype=torch.float16)},
                "4d03": {cached_randn((1, 3, 4, 1), dtype=torch.float16)},
                "4d12": {cached_randn((2, 1, 1, 1), dtype=torch.float16)},
                "4d13": {cached_randn((2, 1, 4, 1), dtype=torch.float16)},
                "4d23": {cached_randn((2, 3, 1, 1), dtype=torch.float16)},
                "4d012": {cached_randn((1, 1, 1, 5), dtype=torch.float16)},
                "4d013": {cached_randn((1, 1, 4, 1), dtype=torch.float16)},
                "4d023": {cached_randn((1, 3, 1, 1), dtype=torch.float16)},
                "4d123": {cached_randn((2, 1, 1, 1), dtype=torch.float16)},
            },
        },
        (
            "test_bitwise_not",
            "test_fallback_unary_op_cpu",
        ): {
            "ops_dict": {
                "bitwise_not": torch.bitwise_not,
            },
            "param_sets": {
                "bool_1d": (cached_randn((256), dtype=torch.float16) > 0,),
                "bool_2d": (cached_randn((128, 256), dtype=torch.float16) > 0,),
                "bool_3d": (cached_randn((8, 32, 128), dtype=torch.float16) > 0,),
                "bool_4d": (cached_randn((2, 8, 32, 64), dtype=torch.float16) > 0,),
                "int_1d": (torch.randint(-128, 127, (256,), dtype=torch.int8),),
                "int_2d": (torch.randint(-128, 127, (128, 256), dtype=torch.int8),),
                "int_3d": (torch.randint(-128, 127, (8, 32, 128), dtype=torch.int8),),
                "int_4d": (torch.randint(-128, 127, (2, 8, 32, 64), dtype=torch.int8),),
            },
        },
        (
            "test_bitwise_and",
            "test_fallback_binary_op_cpu",
        ): {
            "ops_dict": {
                "bitwise_and": torch.bitwise_and,
            },
            "param_sets": {
                "bool_1d": (
                    cached_randn((256), dtype=torch.float16) > 0,
                    cached_randn((256), dtype=torch.float16) > 0,
                ),
                "bool_2d": (
                    cached_randn((128, 256), dtype=torch.float16) > 0,
                    cached_randn((128, 256), dtype=torch.float16) > 0,
                ),
                "bool_3d": (
                    cached_randn((8, 32, 128), dtype=torch.float16) > 0,
                    cached_randn((8, 32, 128), dtype=torch.float16) > 0,
                ),
                "bool_4d": (
                    cached_randn((2, 8, 32, 64), dtype=torch.float16) > 0,
                    cached_randn((2, 8, 32, 64), dtype=torch.float16) > 0,
                ),
                "int_1d": (
                    torch.randint(-128, 127, (256,), dtype=torch.int8),
                    torch.randint(-128, 127, (256,), dtype=torch.int8),
                ),
                "int_2d": (
                    torch.randint(-128, 127, (128, 256), dtype=torch.int8),
                    torch.randint(-128, 127, (128, 256), dtype=torch.int8),
                ),
                "int_3d": (
                    torch.randint(-128, 127, (8, 32, 128), dtype=torch.int8),
                    torch.randint(-128, 127, (8, 32, 128), dtype=torch.int8),
                ),
                "int_4d": (
                    torch.randint(-128, 127, (2, 8, 32, 64), dtype=torch.int8),
                    torch.randint(-128, 127, (2, 8, 32, 64), dtype=torch.int8),
                ),
            },
        },
        (
            "test_logical_not",
            "test_fallback_unary_op_cpu",
        ): {
            "ops_dict": {
                "logical_not": torch.logical_not,
            },
            "param_sets": {
                "1d_fp16": (cached_randn(128, dtype=torch.float16),),
                "1d_bool": (cached_randn(128, dtype=torch.float16) > 0,),
                "2d_fp16": (cached_randn((4, 128), dtype=torch.float16),),
                "2d_bool": (cached_randn((4, 128), dtype=torch.float16) > 0,),
                "3d_fp16": (cached_randn((2, 4, 128), dtype=torch.float16),),
                "3d_bool": (cached_randn((2, 4, 128), dtype=torch.float16) > 0,),
                "4d_fp16": (cached_randn((1, 2, 4, 128), dtype=torch.float16),),
                "4d_bool": (cached_randn((1, 2, 4, 128), dtype=torch.float16) > 0,),
                "fp16_single_elem": (cached_randn(1, dtype=torch.float16),),
                "bool_single_elem": (cached_randn(1, dtype=torch.float16) > 0,),
                # TODO: Fix torch.eq(-0.0,0.0) equality bug (Issue 628)
                # "fp16_signed_0": (torch.tensor([0.0, -0.0, 1.0, -1.0], dtype=torch.float16),),
            },
        },
        (
            "test_inplace_op",
            "test_inplace_op_cpu",
        ): {
            "ops_dict": {
                "add": torch.Tensor.add_,
                "mul": torch.Tensor.mul_,
            },
            "param_sets": {
                "1d": (
                    torch.zeros(128, dtype=torch.float16),
                    cached_randn((128,)),
                ),
                "2d": (
                    torch.zeros(4, 128, dtype=torch.float16),
                    cached_randn((4, 128)),
                ),
                "3d": (
                    torch.zeros(3, 4, 128, dtype=torch.float16),
                    cached_randn((3, 4, 128)),
                ),
            },
        },
        (
            "test_inplace_copy",
            "test_inplace_op_cpu",
        ): {
            "ops_dict": {
                "copy": torch.Tensor.copy_,
            },
            "param_sets": {
                "1d": (
                    torch.zeros(128, dtype=torch.float16),
                    cached_randn((128,)),
                ),
                "2d": (
                    torch.zeros(4, 128, dtype=torch.float16),
                    cached_randn((4, 128)),
                ),
                "3d": (
                    torch.zeros(3, 4, 128, dtype=torch.float16),
                    cached_randn((3, 4, 128)),
                ),
                "bool": (
                    torch.zeros(128, dtype=torch.bool),  # bool tensor
                    (cached_randn((128,)) > 0),  # bool tensor
                ),
                # TODO: Copying bool tensors to host is not working yet. See issue #488.
                # "float2bool": (
                #     torch.zeros(128, dtype=torch.bool),  # bool tensor
                #     (cached_randn((128,)) > 0).to(dtype=torch.float16),  # float tensor
                # ),
                "bool2float": (
                    torch.zeros(128, dtype=torch.float16),  # float tensor
                    cached_randn((128,)) > 0,  # bool tensor
                ),
                "2d_transposed_src": (
                    torch.zeros(128, 256, dtype=torch.float16),
                    cached_randn((256, 128)).t(),
                ),
            },
        },
        (
            "test_inplace_copy_noncontiguous",
            "test_inplace_copy_noncontiguous_cpu",
        ): {
            "param_sets": {
                "transposed_dst": (
                    torch.zeros(256, 128, dtype=torch.float16),
                    cached_randn((128, 256)),
                ),
                "transposed_src_and_dst": (
                    torch.zeros(256, 128, dtype=torch.float16),
                    cached_randn((256, 128)).t(),
                ),
            },
        },
        (
            "test_squeeze",
            "test_dim_op_cpu_eager",
        ): {
            "ops_dict": {
                "single": lambda dim, x: torch.squeeze(x, dim),
            },
            "param_sets": {
                "2d0": (0, cached_randn((1, 128))),
                "2d1": (1, cached_randn((4, 1))),
                "3d0": (0, cached_randn((1, 4, 128))),
                "3d1": (1, cached_randn((3, 1, 128))),
                "3d2": (2, cached_randn((3, 4, 1))),
                "4d0": (0, cached_randn((1, 3, 4, 128))),
                "4d1": (1, cached_randn((2, 1, 4, 128))),
                "4d2": (2, cached_randn((2, 3, 1, 128))),
                "4d3": (3, cached_randn((2, 3, 4, 1))),
            },
        },
        (
            "test_squeeze",
            "test_dim_op_cpu",
        ): {
            "ops_dict": {
                # exp(squeeze(x)) triggers internal compile in eager mode that
                # fails on shapes where the squeezed dim is the last dimension
                "combined": lambda dim, x: torch.exp(torch.squeeze(x, dim)),
            },
            "param_sets": {
                "2d0": (0, cached_randn((1, 128))),
                "2d1": (1, cached_randn((4, 1))),
                "3d0": (0, cached_randn((1, 4, 128))),
                "3d1": (1, cached_randn((3, 1, 128))),
                "3d2": (2, cached_randn((3, 4, 1))),
                "4d0": (0, cached_randn((1, 3, 4, 128))),
                "4d1": (1, cached_randn((2, 1, 4, 128))),
                "4d2": (2, cached_randn((2, 3, 1, 128))),
                "4d3": (3, cached_randn((2, 3, 4, 1))),
            },
        },
        (
            "test_squeeze_reduction",
            "test_dim_op_cpu_eager",
        ): {
            "ops_dict": {
                "sum": lambda dim, x: torch.squeeze(
                    torch.sum(x, dim, keepdim=True), dim
                ),
            },
            "param_sets": {
                "2d0": (0, cached_randn((4, 128))),
                "3d0": (0, cached_randn((3, 4, 128))),
                "3d1": (1, cached_randn((3, 4, 128))),
                "4d0": (0, cached_randn((2, 3, 4, 128))),
                "4d1": (1, cached_randn((2, 3, 4, 128))),
                "4d2": (2, cached_randn((2, 3, 4, 128))),
                # TODO: Support sparse tensors
                # "3d2": (2, cached_randn((3, 4, 128))),
                # "2d1": (1, cached_randn((4, 128))),
                # "4d3": (3, cached_randn((2, 3, 4, 128))),
            },
        },
        (
            "test_unsqueeze",
            "test_dim_op_cpu_eager",
        ): {
            "ops_dict": {
                "single": lambda dim, x: torch.unsqueeze(x, dim),
            },
            "param_sets": {
                "1d0": (0, cached_randn((128,))),
                "1d1": (1, cached_randn((128,))),
                "2d0": (0, cached_randn((4, 128))),
                "2d1": (1, cached_randn((4, 128))),
                "2d2": (2, cached_randn((4, 128))),
                "3d0": (0, cached_randn((3, 4, 128))),
                "3d1": (1, cached_randn((3, 4, 128))),
                "3d2": (2, cached_randn((3, 4, 128))),
                "3d3": (3, cached_randn((3, 4, 128))),
                "4d0": (0, cached_randn((2, 3, 4, 128))),
                "4d1": (1, cached_randn((2, 3, 4, 128))),
                "4d2": (2, cached_randn((2, 3, 4, 128))),
                "4d3": (3, cached_randn((2, 3, 4, 128))),
                "4d4": (4, cached_randn((2, 3, 4, 128))),
            },
        },
        (
            "test_unsqueeze",
            "test_dim_op_cpu",
        ): {
            "ops_dict": {
                # exp(unsqueeze(x)) triggers internal compile in eager mode that
                # fails with host dimension lookup errors
                "combined": lambda dim, x: torch.exp(torch.unsqueeze(x, dim)),
            },
            "param_sets": {
                "1d0": (0, cached_randn((128,))),
                "1d1": (1, cached_randn((128,))),
                "2d0": (0, cached_randn((4, 128))),
                "2d1": (1, cached_randn((4, 128))),
                "2d2": (2, cached_randn((4, 128))),
                "3d0": (0, cached_randn((3, 4, 128))),
                "3d1": (1, cached_randn((3, 4, 128))),
                "3d2": (2, cached_randn((3, 4, 128))),
                "3d3": (3, cached_randn((3, 4, 128))),
                "4d0": (0, cached_randn((2, 3, 4, 128))),
                "4d1": (1, cached_randn((2, 3, 4, 128))),
                "4d2": (2, cached_randn((2, 3, 4, 128))),
                "4d3": (3, cached_randn((2, 3, 4, 128))),
                "4d4": (4, cached_randn((2, 3, 4, 128))),
            },
        },
        (
            "test_unsqueeze_broadcast",
            "test_dim_op_cpu",
        ): {
            "ops_dict": {
                "add": lambda dim, x, y: torch.add(x, torch.unsqueeze(y, dim)),
            },
            "param_sets": {
                "1d0": (0, cached_randn((4, 128)), cached_randn((128,))),
                "2d0": (0, cached_randn((3, 4, 128)), cached_randn((4, 128))),
                "2d1": (1, cached_randn((3, 4, 128)), cached_randn((3, 128))),
                "3d0": (0, cached_randn((2, 3, 4, 128)), cached_randn((3, 4, 128))),
                "3d1": (1, cached_randn((2, 3, 4, 128)), cached_randn((2, 4, 128))),
                "3d2": (2, cached_randn((2, 3, 4, 128)), cached_randn((2, 3, 128))),
                # TODO: Support dim=-1 for broadcasting. See: #598
                # "1d1": (1, cached_randn((4, 128)), cached_randn((4,))),
                # "2d2": (2, cached_randn((3, 4, 128)), cached_randn((3, 4))),
                # "3d3": (3, cached_randn((2, 3, 4, 128)), cached_randn((2, 3, 4))),
            },
        },
        ("test_attention", "test_attention_cpu"): {
            "param_sets": {
                "3d": (
                    cached_randn((4, 256, 128), dtype=torch.float16),  # q
                    cached_randn((4, 256, 128), dtype=torch.float16),  # k
                    cached_randn((4, 256, 128), dtype=torch.float16),  # v
                    torch.tensor(1 / (128**0.5), dtype=torch.float16).repeat(
                        4, 256, 256
                    ),  # sm_scale
                ),
                "3d_batch_size_1": (
                    cached_randn((1, 4, 256, 128), dtype=torch.float16),  # q
                    cached_randn((1, 4, 256, 128), dtype=torch.float16),  # k
                    cached_randn((1, 4, 256, 128), dtype=torch.float16),  # v
                    torch.tensor(1 / (128**0.5), dtype=torch.float16).repeat(
                        4, 256, 256
                    ),  # sm_scale
                ),
                "4d": (
                    cached_randn((8, 4, 128, 64), dtype=torch.float16),  # q
                    cached_randn((8, 4, 128, 64), dtype=torch.float16),  # k
                    cached_randn((8, 4, 128, 64), dtype=torch.float16),  # v
                    torch.tensor(1 / (128**0.5), dtype=torch.float16).repeat(
                        8, 4, 128, 128
                    ),  # sm_scale
                ),
            },
        },
        ("test_layernorm", "test_layernorm_cpu"): {
            "param_sets": {
                "2d": (
                    cached_randn((256, 128), dtype=torch.float16),  # input
                    cached_randn((128), dtype=torch.float16),  # weight
                    torch.zeros([128], dtype=torch.float16),  # bias
                ),
            },
        },
        ("test_rmsnorm", "test_rmsnorm_cpu"): {
            "param_sets": {
                "2d": (cached_randn((256, 128), dtype=torch.float16),),
                "3d": (cached_randn((64, 256, 128), dtype=torch.float16),),
                "4d": (cached_randn((4, 17, 256, 128), dtype=torch.float16),),
            },
        },
        ("test_softplus", "test_softplus_cpu"): {
            "param_sets": {
                "2d": (cached_randn((256, 128), dtype=torch.float16),),
                "3d": (cached_randn((64, 256, 128), dtype=torch.float16),),
                "4d": (cached_randn((4, 17, 256, 128), dtype=torch.float16),),
            },
        },
        # --- Migrated from test_ops.py ---
        ("test_copy_roundtrip", "test_copy_roundtrip"): {
            "param_sets": {
                # Aligned shapes
                "1d": (cached_randn((256,), dtype=torch.float16),),
                "2d": (cached_randn((256, 128), dtype=torch.float16),),
                "3d": (cached_randn((256, 128, 512), dtype=torch.float16),),
                "4d": (cached_randn((2, 6, 3, 128), dtype=torch.float16),),
                "5d": (cached_randn((4, 8, 3, 64, 256), dtype=torch.float16),),
                "6d": (cached_randn((4, 8, 16, 12, 64, 128), dtype=torch.float16),),
                # Padded (non-stick-aligned last dim)
                "1d_padded": (cached_randn((511,), dtype=torch.float16),),
                "2d_padded": (cached_randn((2, 205), dtype=torch.float16),),
                "3d_padded": (cached_randn((2, 2, 72), dtype=torch.float16),),
                "4d_padded": (cached_randn((2, 2, 2, 120), dtype=torch.float16),),
                # Small tensors requiring stick padding
                "1d_stick": (torch.tensor([1, 2, 3], dtype=torch.float16),),
                "2d_stick": (
                    torch.tensor([[1, -2, 3], [4, 5, 6]], dtype=torch.float16),
                ),
                "3d_stick": (
                    torch.tensor(
                        [[[1, -2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                        dtype=torch.float16,
                    ),
                ),
                "4d_stick": (torch.rand(2, 2, 2, 3, dtype=torch.float16),),
                "5d_stick": (torch.rand(1, 2, 3, 4, 5, dtype=torch.float16),),
                "6d_stick": (torch.rand(1, 3, 5, 2, 4, 62, dtype=torch.float16),),
            },
        },
        ("test_mean", "test_mean_cpu"): {
            "param_sets": {
                "3d_dim0": (
                    0,
                    False,
                    torch.tensor(
                        [
                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                        ],
                        dtype=torch.float16,
                    ),
                ),
                "3d_dim1": (
                    1,
                    False,
                    torch.tensor(
                        [
                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                        ],
                        dtype=torch.float16,
                    ),
                ),
                "3d_dim0_keepdim": (
                    0,
                    True,
                    torch.tensor(
                        [
                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                        ],
                        dtype=torch.float16,
                    ),
                ),
            },
        },
        ("test_zeros", "test_zeros_cpu"): {
            "param_sets": {
                "aligned": ((3, 64),),
                "padded": ((3, 50),),
            },
        },
        ("test_fill_scalar", "test_fill_scalar_cpu"): {
            "param_sets": {
                "1d_eager": (
                    5.0,
                    torch.tensor([1, -2, 3], dtype=torch.float16),
                    "eager",
                ),
                "1d_compiled": (
                    5.0,
                    torch.tensor([1, -2, 3], dtype=torch.float16),
                    "compiled",
                ),
            },
        },
        ("test_addmm_scaled", "test_addmm_scaled_cpu"): {
            "param_sets": {
                "alpha_0_5": (
                    0.5,
                    cached_randn((67, 128), dtype=torch.float16),
                    cached_randn((67, 256), dtype=torch.float16),
                    cached_randn((256, 128), dtype=torch.float16),
                ),
            },
        },
        ("test_addmm_out", "test_addmm_out_cpu"): {
            "param_sets": {
                "basic": (
                    cached_randn((67, 128), dtype=torch.float16),
                    cached_randn((67, 256), dtype=torch.float16),
                    cached_randn((256, 128), dtype=torch.float16),
                ),
            },
        },
        ("test_embedding", "test_embedding_cpu"): {
            "param_sets": {
                "basic": (
                    torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.int64),
                    torch.rand(10, 3, dtype=torch.float16),
                    None,
                ),
                "padding_idx": (
                    torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.int64),
                    torch.rand(10, 3, dtype=torch.float16),
                    0,
                ),
            },
        },
        ("test_isin", "test_isin_cpu"): {
            "param_sets": {
                "tensor_tensor": (
                    torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64),
                    torch.tensor([2, 4], dtype=torch.int64),
                ),
            },
        },
        ("test_isin_out", "test_isin_out_cpu"): {
            "param_sets": {
                "tensor_tensor": (
                    torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64),
                    torch.tensor([2, 4], dtype=torch.int64),
                ),
            },
        },
        ("test_scalar_cpu", "test_scalar_cpu"): {
            "ops_dict": {
                "add": torch.add,
                "sub": torch.sub,
                "mul": torch.mul,
                "div": torch.div,
                "true_divide": torch.true_divide,
                "combined": lambda scalar, x: (
                    a := torch.add(x, scalar),
                    b := torch.add(scalar, a),
                    c := torch.add(b, scalar),
                    d := torch.sub(c, scalar),
                    e := torch.mul(5, d),
                    out := torch.add(e, e),
                    out,
                ),
            },
            "param_sets": {
                "1d": (cached_randn((1024,), dtype=torch.float16), 3.0),
                "2d": (cached_randn((512, 1024), dtype=torch.float16), 1.0),
                "3d": (cached_randn((8, 64, 1024), dtype=torch.float16), 1.5),
                "4d": (cached_randn((2, 4, 64, 1024), dtype=torch.float16), 2.4),
            },
        },
        ("test_linear", "test_linear_fn"): {
            "param_sets": {
                "2d_no_bias": (
                    cached_randn((67, 256)),
                    cached_randn((128, 256)),
                    None,
                ),
                "2d_bias": (
                    cached_randn((67, 256)),
                    cached_randn((128, 256)),
                    cached_randn((128,)),
                ),
                "3d_no_bias": (
                    cached_randn((3, 17, 256)),
                    cached_randn((128, 256)),
                    None,
                ),
                "3d_bias": (
                    cached_randn((3, 17, 256)),
                    cached_randn((128, 256)),
                    cached_randn((128,)),
                ),
            }
        },
        ("test_tril", "test_tril_cpu"): {
            "param_sets": {
                "2d": (cached_randn((64, 64)),),
                "3d": (cached_randn((32, 64, 64)),),
            }
        },
        ("test_triu", "test_triu_cpu"): {
            "param_sets": {
                "2d": (
                    cached_randn((64, 64)),
                    1,
                ),
                "3d": (
                    cached_randn((32, 64, 64)),
                    1,
                ),
            }
        },
        ("test_item", "test_item_cpu"): {
            "param_sets": {
                "float16": (torch.tensor([3.14], dtype=torch.float16),),
                "float32": (torch.tensor([2.71828], dtype=torch.float32),),
                "scalar_float": (torch.tensor(3.14, dtype=torch.float32),),
                "int64": (torch.tensor([5], dtype=torch.int64),),
                "from_computation": (
                    torch.tensor([2.0], dtype=torch.float16),
                    torch.tensor([3.0], dtype=torch.float16),
                ),
            },
        },
        ("test_sdpa", "test_sdpa_cpu"): {
            "param_sets": {
                "mha_prefill": (
                    cached_randn(
                        (2, 256, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 32, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 32, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    None,
                    False,
                    False,
                ),
                "mha_prefill_causal": (
                    cached_randn(
                        (2, 256, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 32, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 32, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    None,
                    True,
                    False,
                ),
                "mha_prefill_mask": (
                    cached_randn(
                        (2, 256, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 32, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 32, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    torch.triu(
                        torch.ones((256, 256), dtype=torch.float16) * -float("inf"),
                        diagonal=1,
                    ),
                    False,
                    False,
                ),
                "gqa_prefill": (
                    cached_randn(
                        (2, 256, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 8, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 8, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    None,
                    False,
                    True,
                ),
                "gqa_prefill_causal": (
                    cached_randn(
                        (2, 256, 32, 128), differentiation=1, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 8, 128), differentiation=2, dtype=torch.float16
                    ).transpose(1, 2),
                    cached_randn(
                        (2, 256, 8, 128), differentiation=3, dtype=torch.float16
                    ).transpose(1, 2),
                    None,
                    True,
                    True,
                ),
                # TODO(aviros): Implement broadcast for batch dim in batch matmul
                # "mha_decode": (
                #     cached_randn(
                #         (2, 1, 32, 128), differentiation=1, dtype=torch.float16
                #     ),
                #     cached_randn(
                #         (2, 257, 32, 128), differentiation=2, dtype=torch.float16
                #     ),
                #     cached_randn(
                #         (2, 257, 32, 128), differentiation=3, dtype=torch.float16
                #     ),
                #     False,
                #     False,
                # ),
                # TODO(aviros): Implement broadcast for batch dim in batch matmul, expand
                # "gqa_decode": (
                #     cached_randn(
                #         (2, 1, 32, 128), differentiation=1, dtype=torch.float16
                #     ),
                #     cached_randn(
                #         (2, 257, 8, 128), differentiation=2, dtype=torch.float16
                #     ),
                #     cached_randn(
                #         (2, 257, 8, 128), differentiation=3, dtype=torch.float16
                #     ),
                #     False,
                #     True,
                # ),
            }
        },
        ("test_split", "test_split_cpu"): {
            "ops_dict": {
                "split3": lambda dim, index, x: (
                    torch.split(x, x.size()[dim] // 3, dim=dim)[index].clone(),
                ),
            },
            "param_sets": {
                "1d0s0": (0, 0, cached_randn((384,), dtype=torch.float16)),
                "1d0s1": (0, 1, cached_randn((384,), dtype=torch.float16)),
                "1d0s2": (0, 2, cached_randn((384,), dtype=torch.float16)),
                "2d0s0": (0, 0, cached_randn((9, 384), dtype=torch.float16)),
                "2d0s1": (0, 1, cached_randn((9, 384), dtype=torch.float16)),
                "2d0s2": (0, 2, cached_randn((9, 384), dtype=torch.float16)),
                "2d1s0": (1, 0, cached_randn((9, 384), dtype=torch.float16)),
                "2d1s1": (1, 1, cached_randn((9, 384), dtype=torch.float16)),
                "2d1s2": (1, 2, cached_randn((9, 384), dtype=torch.float16)),
                "3d0s0": (0, 0, cached_randn((9, 15, 384), dtype=torch.float16)),
                "3d0s1": (0, 1, cached_randn((9, 15, 384), dtype=torch.float16)),
                "3d0s2": (0, 2, cached_randn((9, 15, 384), dtype=torch.float16)),
                "3d1s0": (1, 0, cached_randn((9, 15, 384), dtype=torch.float16)),
                "3d1s1": (1, 1, cached_randn((9, 15, 384), dtype=torch.float16)),
                "3d1s2": (1, 2, cached_randn((9, 15, 384), dtype=torch.float16)),
                "3d2s0": (2, 0, cached_randn((9, 15, 384), dtype=torch.float16)),
                "3d2s1": (2, 1, cached_randn((9, 15, 384), dtype=torch.float16)),
                "3d2s2": (2, 2, cached_randn((9, 15, 384), dtype=torch.float16)),
            },
        },
        ("test_slice", "test_slice_cpu"): {
            "ops_dict": {
                "slice3": lambda dim, index, x: x.exp(),
            },
            "param_sets": {
                # TODO: Add more tests by generalizing size-1 dim support. See #1548
                "3d1s0": (1, 0, cached_randn((5, 3, 192), dtype=torch.float16)),
                "3d1s1": (1, 1, cached_randn((5, 3, 192), dtype=torch.float16)),
                "3d1s2": (1, 2, cached_randn((5, 3, 192), dtype=torch.float16)),
            },
        },
        ("test_rope_fms", "test_rope_cpu"): {
            "param_sets": {
                "prefill_bs1": (
                    cached_randn((1, 256, 4096), dtype=torch.float16),
                    cached_randn((1, 256, 2, 2, 64), dtype=torch.float16),
                ),
                "prefill": (
                    cached_randn((2, 256, 4096), dtype=torch.float16),
                    cached_randn((1, 256, 2, 2, 64), dtype=torch.float16),
                ),
                "decode_bs1": (
                    cached_randn((1, 1, 4096), dtype=torch.float16),
                    cached_randn((1, 1, 2, 2, 64), dtype=torch.float16),
                ),
                "decode": (
                    cached_randn((2, 1, 4096), dtype=torch.float16),
                    cached_randn((1, 1, 2, 2, 64), dtype=torch.float16),
                ),
            },
        },
        ("test_qkv_attn_paths_fms", "test_attn_qkv_paths"): {
            "param_sets": {
                "prefill_mha": (
                    cached_randn(
                        (1, 256, 32, 2, 1, 64), differentiation=1, dtype=torch.bfloat16
                    ),
                    cached_randn(
                        (1, 256, 32, 2, 1, 64), differentiation=2, dtype=torch.bfloat16
                    ),
                    cached_randn((1, 256, 4096), dtype=torch.bfloat16),
                ),
                "prefill_gqa": (
                    cached_randn((1, 256, 32, 2, 1, 64), dtype=torch.bfloat16),
                    cached_randn((1, 256, 8, 2, 1, 64), dtype=torch.bfloat16),
                    cached_randn((1, 256, 1024), dtype=torch.bfloat16),
                ),
                "fms_decode_mha": (
                    cached_randn((1, 64, 32, 2, 1, 64), dtype=torch.bfloat16),
                    cached_randn((1, 320, 32, 2, 1, 64), dtype=torch.bfloat16),
                    cached_randn((1, 320, 4096), dtype=torch.bfloat16),
                ),
                "fms_decode_gqa": (
                    cached_randn((1, 64, 32, 2, 1, 64), dtype=torch.bfloat16),
                    cached_randn((1, 320, 8, 2, 1, 64), dtype=torch.bfloat16),
                    cached_randn((1, 320, 1024), dtype=torch.bfloat16),
                ),
                "decode_mha": (
                    cached_randn((1, 1, 32, 2, 1, 64), dtype=torch.bfloat16),
                    cached_randn((1, 257, 32, 2, 1, 64), dtype=torch.bfloat16),
                    cached_randn((1, 257, 4096), dtype=torch.bfloat16),
                ),
                "decode_gqa": (
                    cached_randn((1, 1, 32, 2, 1, 64), dtype=torch.bfloat16),
                    cached_randn((1, 257, 8, 2, 1, 64), dtype=torch.bfloat16),
                    cached_randn((1, 257, 1024), dtype=torch.bfloat16),
                ),
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
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
        dtype = torch.bool
        x = torch.randint(0, 2, (2, 64), dtype=dtype)
        x_spyre = x.to("spyre")
        y = torch.randint(0, 2, (2, 64), dtype=dtype)
        y_spyre = y.to("spyre")
        result = torch.compile(torch.eq, dynamic=False)(x_spyre, y_spyre).cpu()
        torch.testing.assert_close(result, torch.eq(x, y))

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_scalar_cpu(self, op, *args):
        def fn(*tensor_args):
            # Scalar args are preserved as scalars
            tensor_args = list(tensor_args)
            updated_args = [
                tensor_args.pop(0) if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ]
            return op(*updated_args)

        tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]

        compare_with_cpu(fn, *tensor_args)

    def test_unary_op_cpu(self, op, x):
        compare_with_cpu(op, x)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_fallback_unary_op_cpu(self, op, x):
        compare_with_cpu(op, x)

    def test_binary_op(self, op, a, b):
        if op == torch.div:
            # TODO: Division by 0 or near-zero differs on Spyre from CPU, sidestep for now.
            tiny_value_mask = torch.abs(b) < FP16_EPS
            b[tiny_value_mask] = FP16_EPS

        if a.dtype == torch.float32:
            compare_with_cpu(op, a, b)
        else:
            compare(op, a, b)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_fallback_binary_op_cpu(self, op, x, y):
        compare_with_cpu(op, x, y, run_eager=False)

    # Increased mm test tolerance for splitk
    def test_mm_relaxed(self, op, a, b):
        K = b.shape[-2]
        if K > (128 // b.element_size()):  # multiple sticks
            compare(op, a, b, atol=0.1, rtol=0.1)
        else:  # single stick, no need to relax
            compare(op, a, b)

    def test_binary_op_cpu(self, op, x, y):
        # Eager mode support varies by op:
        # - torch.eq, torch.ge, torch.gt, torch.lt: work eagerly
        # - torch.ne, torch.le: aten::ne.Tensor_out / aten::le.Tensor_out not registered
        # - torch.matmul: numerical divergence (close=False) in eager 2d case
        eager_supported = op in (torch.eq, torch.ge, torch.gt, torch.lt)
        compare_with_cpu(op, x, y, run_eager=eager_supported)

    def test_linear_fn(self, x, weight, bias):
        # NOTE: relaxing atol from 2e-1 to 3e-1 for multi-dim work division, single element fails without
        compare_with_cpu(
            torch.nn.functional.linear, x, weight, bias, atol=3e-1, rtol=2e-1
        )

    @unittest.skip("deeptools: error")
    def test_add_broadcast(self, x, y):
        compare(lambda x, y: torch.add(x[None, :], y), x, y)

    # Example where base function is not parameterized
    def test_add_broadcast_cpu(self, x, y):
        compare_with_cpu(lambda x, y: torch.add(x[None, :], y), x, y)

    def test_addmm_cpu(self, input, mat1, mat2):
        # NOTE: relaxing atol from 2e-1 to 3e-1 for multi-dim work division
        compare_with_cpu(torch.addmm, input, mat1, mat2, atol=3e-1, rtol=2e-1)

    # @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    # @pytest.mark.filterwarnings("ignore:Backend Spyre does not support int64")
    # def test_reduce_cpu(self, op, x):
    #    compare_with_cpu(lambda x: op(x), x)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings("ignore:Backend Spyre does not support int64")
    def test_reduce_keepdim0_cpu(self, op, dim: int, x):
        # torch.max returns a tuple; torch.amax is not registered for Spyre eager dispatch
        if op == torch.amax or op == torch.max:
            # aten::amax.out is not registered for the Spyre backend
            compare_with_cpu(
                lambda x: op(x, dim=dim, keepdim=False), x, run_eager=False
            )
        else:
            compare_with_cpu(lambda x: op(x, dim=dim, keepdim=False), x)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings("ignore:Backend Spyre does not support int64")
    def test_reduce_keepdim0_cpu_no_eager(self, op, dim: int, x):
        # aten::max.dim and aten::amin are not registered for Spyre eager dispatch
        compare_with_cpu(lambda x: op(x, dim=dim, keepdim=False), x, run_eager=False)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings("ignore:Backend Spyre does not support int64")
    def test_reduce_keepdim1_cpu(self, op, dim: int, x):
        if op == torch.amax or op == torch.max:
            # aten::amax.out is not registered for the Spyre backend
            compare_with_cpu(lambda x: op(x, dim=dim, keepdim=True), x, run_eager=False)
        else:
            compare_with_cpu(lambda x: op(x, dim=dim, keepdim=True), x)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings("ignore:Backend Spyre does not support int64")
    def test_reduce_keepdim1_cpu_no_eager(self, op, dim: int, x):
        # aten::max.dim and aten::amin are not registered for Spyre eager dispatch
        if op == torch.max:
            compare_with_cpu(
                lambda x: op(x, dim=dim, keepdim=True)[0], x, run_eager=False
            )
        else:
            compare_with_cpu(lambda x: op(x, dim=dim, keepdim=True), x, run_eager=False)

    def test_max_sub_broadcast(self, dim: int, x):
        def fn(x):
            x_max = torch.max(x, dim=dim).values
            z = x - torch.unsqueeze(x_max, dim=dim)
            return z

        compare(fn, x)

    def test_t_1d_cpu(self, x):
        compare_with_cpu(lambda x: x.t(), x)

    def test_t_1d_contiguous_cpu(self, x):
        # Note: .contiguous() causes issues with eager mode, see https://github.com/torch-spyre/torch-spyre/issues/1149
        compare_with_cpu(lambda x: x.t().contiguous(), x, run_eager=False)

    def test_t_2d_cpu(self, x):
        compare_with_cpu(lambda x: x.t(), x)

    def test_t_2d_contiguous_cpu(self, x):
        # Note: .contiguous() causes issues with eager mode, see https://github.com/torch-spyre/torch-spyre/issues/1149
        compare_with_cpu(lambda x: x.t().contiguous(), x, run_eager=False)

    def test_transpose_2d_cpu(self, dim0: int, dim1: int, x):
        compare_with_cpu(lambda x: torch.transpose(x, dim0, dim1), x)

    def test_transpose_2d_contiguous_cpu(self, dim0: int, dim1: int, x):
        # Note: .contiguous() causes issues with eager mode, see https://github.com/torch-spyre/torch-spyre/issues/1149
        compare_with_cpu(
            lambda x: torch.transpose(x, dim0, dim1).contiguous(), x, run_eager=False
        )

    def test_transpose_3d_cpu(self, dim0: int, dim1: int, x):
        compare_with_cpu(lambda x: torch.transpose(x, dim0, dim1), x)

    def test_transpose_3d_contiguous_cpu(self, dim0: int, dim1: int, x):
        # Note: .contiguous() causes issues with eager mode, see https://github.com/torch-spyre/torch-spyre/issues/1149
        compare_with_cpu(
            lambda x: torch.transpose(x, dim0, dim1).contiguous(), x, run_eager=False
        )

    def test_transpose_4d_cpu(self, dim0: int, dim1: int, x):
        compare_with_cpu(lambda x: torch.transpose(x, dim0, dim1), x)

    def test_transpose_4d_contiguous_cpu(self, dim0: int, dim1: int, x):
        # Note: .contiguous() causes issues with eager mode, see https://github.com/torch-spyre/torch-spyre/issues/1149
        compare_with_cpu(
            lambda x: torch.transpose(x, dim0, dim1).contiguous(), x, run_eager=False
        )

    def test_where_cpu(self, cond_op, x, y):
        # aten::where.self is not registered for the Spyre backend
        compare_with_cpu(
            lambda x, y: torch.where(cond_op(x, y), x, y), x, y, run_eager=False
        )

    def test_range_op(self, op, input, min, max, err):
        # aten::clamp is not registered for Spyre eager dispatch; it uses the
        # spyre::clamp custom op which only works inside torch.compile
        compare_with_cpu(
            lambda x: op(x, min, max), input, atol=err, rtol=err, run_eager=False
        )

    def test_activation_cls(self, op, input, kwargs, err):
        # Spyre activation custom ops (e.g. spyre::gelu) have a pass-through
        # implementation that returns None in eager mode; they only work inside
        # torch.compile where the inductor lowering handles them
        compare_with_cpu(
            lambda x: op(**kwargs)(x), input, atol=err, rtol=err, run_eager=False
        )

    def test_activation_fn(self, op, input, err):
        compare_with_cpu(lambda x: op(x), input, atol=err, rtol=err)

    @pytest.mark.filterwarnings(
        "ignore:Backend Spyre does not support int64:UserWarning"
    )
    def test_clone(self, x):
        # Eager clone + .cpu() causes heap corruption (invalid fastbin / corrupted
        # double-linked list) in libsenlib for fp16/fp32 small tensors, and SIGBUS
        # for bool tensors.  Disable eager mode for all dtypes.
        compare_with_cpu(lambda a: torch.clone(a).contiguous(), x, run_eager=False)

    def test_permute(self, input_dims, dims):
        compare_with_cpu(
            lambda input: torch.permute(input, dims),
            cached_randn(input_dims, dtype=torch.float16),
        )

    def test_dropout_functional(self, input, kwargs):
        compare_with_cpu(lambda a: torch.nn.functional.dropout(a, **kwargs), input)

    def test_inplace_op_cpu(self, op, dst, src):
        def fn(dst, src):
            dst = dst.clone()
            result = op(dst, src)
            assert id(result) == id(dst)
            return result

        # Eager mode hangs/crashes when executing inplace operations on Spyre tensors
        compare_with_cpu(fn, dst, src, run_eager=False)

    def test_inplace_copy_noncontiguous_cpu(self, dst, src):
        def fn(dst, src):
            dst_t = dst.t()
            dst_t.copy_(src)
            return dst_t.contiguous()

        compare_with_cpu(fn, dst, src, run_eager=False)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_fallback_cpu(self, x):
        def fn(t):
            t = torch.exp(t)  # compiled op
            t = torch.sin(t)  # fallback op
            t = torch.exp(t)  # compiled op
            return t

        with pytest.warns(UserWarning) as record:
            compare_with_cpu(fn, x, cpu_compile=True)

        print(f"Warn {len(record)}")

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_arange_cpu(self, *args):
        def fn(device=None):
            return torch.arange(*args, dtype=torch.float16, device=device)

        compare_with_cpu(fn, needs_device=True)

    def test_empty_like_cpu(self, x):
        def fn(x):
            y = torch.empty_like(x)
            y.fill_(1.0)
            return y

        compare_with_cpu(fn, x)

    @unittest.skip("dtype override may not be supported on Spyre")
    def test_empty_like_dtype_override_cpu(self, x):
        """Test empty_like with dtype override (fp16->fp32 or fp32->fp16)."""
        # Determine target dtype (opposite of input)
        target_dtype = torch.float32 if x.dtype == torch.float16 else torch.float16

        def fn(x):
            y = torch.empty_like(x, dtype=target_dtype)
            y.fill_(1.0)
            return y

        compare_with_cpu(fn, x)

    def test_empty_like_memory_format_cpu(self, x):
        """Test empty_like with memory_format on non-contiguous (transposed) input."""

        def fn(x):
            # Create non-contiguous input via transpose
            x_t = x.t()
            # empty_like with contiguous_format should create contiguous output
            y = torch.empty_like(x_t, memory_format=torch.contiguous_format)
            y.fill_(1.0)
            return y

        # Note: .contiguous() causes issues with eager mode per existing patterns
        compare_with_cpu(fn, x, run_eager=False)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_new_ones_cpu(self, x, y):
        compare_with_cpu(lambda x: x.new_ones((x.size())), x)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_ones_cpu(self, size):
        """Compiled torch.ones(size) on Spyre (identity broadcast) matches CPU."""

        def fn(device=None):
            return torch.ones(size, dtype=torch.float16, device=device)

        compare_with_cpu(fn, needs_device=True, cpu_compile=False)

    def test_numel_cpu(self, x):
        compare_with_cpu(lambda x: torch.numel(x), x)

    def test_cat_cpu(self, dim, *tensors):
        def fn(*tensors):
            return torch.cat(tensors, dim=dim)

        compare_with_cpu(fn, *tensors)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_pad_cpu(self, x, pad):
        """Compiled torch.nn.functional.pad (constant zero) on Spyre matches CPU."""

        def fn(x):
            return torch.nn.functional.pad(x, pad)

        compare_with_cpu(fn, x)

    def test_pad_unsupported(self):
        """Padding cases that raise Unsupported due to logical decomposition constraints."""
        from torch_spyre._inductor.errors import Unsupported

        unsupported_cases = [
            # Negative padding (cropping).
            (cached_randn((3, 64), dtype=torch.float16), (0, -32)),
            (cached_randn((4, 64), dtype=torch.float16), (0, 0, 0, -2)),
        ]
        from torch_spyre._inductor.decompositions import pad_decomp

        for x, pad in unsupported_cases:
            with pytest.raises(Unsupported):
                pad_decomp(x, list(pad))

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_full_cpu(self, *args):
        def fn(device=None):
            return torch.full(*args, dtype=torch.float16, device=device)

        compare_with_cpu(fn, needs_device=True, cpu_compile=False)

    def test_dim_op_cpu(self, op, dim, *args):
        def fn(*args):
            return op(dim, *args)

        # Combined ops (exp+squeeze, exp+unsqueeze, add+unsqueeze) fail in eager
        # because the eager exp/add dispatch internally triggers torch.compile on
        # shapes that the Spyre backend compiler cannot handle
        compare_with_cpu(fn, *args, run_eager=False)

    def test_dim_op_cpu_eager(self, op, dim, *args):
        def fn(*args):
            return op(dim, *args)

        # Simple dim ops (softmax, squeeze, unsqueeze, sum+squeeze) work in eager
        compare_with_cpu(fn, *args)

    def test_attention_cpu(self, *args):
        def fn(q, k, v, sm_scale):
            qk = q @ k.transpose(-1, -2).contiguous()
            p = qk.softmax(dim=-1) * sm_scale
            return p @ v

        # mm/bmm on Spyre tensors segfaults in libsenlib without the torch.compile
        # execution context that normally initialises the hardware session
        compare_with_cpu(fn, *args, run_eager=False)

    def test_layernorm_cpu(self, input, weight, bias):
        def fn(input, weight, bias):
            return torch.nn.functional.layer_norm(
                input, input.shape[1:], weight=weight, bias=bias
            )

        compare_with_cpu(fn, input, weight, bias)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_rmsnorm_cpu(self, x):
        def fn(input):
            return torch.nn.functional.rms_norm(input, [input.shape[-1]], eps=1e-6)

        compare_with_cpu(fn, x)

    def test_softplus_cpu(self, x):
        beta = 1.0
        threshold = 20.0

        def fn(input):
            return torch.nn.functional.softplus(input, beta, threshold)

        compare_with_cpu(fn, x)

    # --- Migrated from test_ops.py ---

    def test_copy_roundtrip(self, x):
        compare_with_cpu(lambda x: x, x)

    def test_mean_cpu(self, dim, keepdim, x):
        compare_with_cpu(lambda x: torch.mean(x, dim=dim, keepdim=keepdim), x)

    def test_zeros_cpu(self, size):
        def fn(device=None):
            return torch.zeros(*size, dtype=torch.float16, device=device)

        compare_with_cpu(fn, needs_device=True, cpu_compile=False)

    def test_fill_scalar_cpu(self, value, x, execution_mode):
        def fn(x):
            x = x.clone()
            x.fill_(value)
            return x

        # RuntimeError: Error: In-device copy not implemented.
        # ISSUE: https://github.com/torch-spyre/torch-spyre/issues/1381
        if execution_mode == "eager":
            pytest.xfail(
                reason="spyre__fill_scalar crashes with SIGBUS in eager mode - in-device copy not implemented"
            )

        compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_addmm_scaled_cpu(self, alpha, input, mat1, mat2):
        compare_with_cpu(
            lambda input, mat1, mat2: torch.addmm(input, mat1, mat2, alpha=alpha),
            input,
            mat1,
            mat2,
            atol=2e-1,
            rtol=2e-1,
        )

    def test_addmm_out_cpu(self, input, mat1, mat2):
        def fn(input, mat1, mat2):
            out = torch.empty(
                mat1.shape[0], mat2.shape[1], dtype=input.dtype, device=input.device
            )
            torch.addmm(input, mat1, mat2, out=out)
            return out

        compare_with_cpu(fn, input, mat1, mat2, atol=2e-1, rtol=2e-1)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_embedding_cpu(self, indices, weight, padding_idx):
        compare_with_cpu(
            lambda indices, weight: torch.nn.functional.embedding(
                indices, weight, padding_idx=padding_idx
            ),
            indices,
            weight,
        )

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_isin_cpu(self, elements, test_elements):
        compare_with_cpu(torch.isin, elements, test_elements)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_isin_out_cpu(self, elements, test_elements):
        def fn(elements, test_elements):
            out = torch.empty(elements.shape, dtype=torch.bool, device=elements.device)
            torch.isin(elements, test_elements, out=out)
            return out

        compare_with_cpu(fn, elements, test_elements)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_isin_tensor_scalar_cpu(self):
        """Test aten.isin.Tensor_Scalar: test_elements is a Python scalar."""
        elements = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        expected = torch.isin(elements, 3)

        elements_spyre = elements.to("spyre")
        actual = torch.isin(elements_spyre, 3).cpu()
        torch.testing.assert_close(actual, expected)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_isin_tensor_scalar_out_cpu(self):
        """Test aten.isin.Tensor_Scalar_out: test_elements is a scalar, out-variant."""
        elements = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        out_cpu = torch.empty(elements.shape, dtype=torch.bool)
        torch.isin(elements, 3, out=out_cpu)

        elements_spyre = elements.to("spyre")
        out_spyre = torch.empty(elements.shape, dtype=torch.bool, device="spyre")
        torch.isin(elements_spyre, 3, out=out_spyre)
        torch.testing.assert_close(out_spyre.cpu(), out_cpu)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_isin_scalar_tensor_cpu(self):
        """Test torch.isin with scalar element and tensor test_elements."""
        test_elements = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        expected = torch.isin(3, test_elements)

        test_elements_spyre = test_elements.to("spyre")
        actual = torch.isin(3, test_elements_spyre).cpu()
        assert actual.item() == expected.item()

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_isin_scalar_tensor_out_cpu(self):
        """Test torch.isin with scalar element, tensor test_elements, and out param."""
        test_elements = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        out_cpu = torch.empty(0, dtype=torch.bool)
        torch.isin(3, test_elements, out=out_cpu)

        test_elements_spyre = test_elements.to("spyre")
        out_spyre = torch.empty((), dtype=torch.bool, device="spyre")
        torch.isin(3, test_elements_spyre, out=out_spyre)
        assert out_spyre.cpu().item() == out_cpu.item()

    def test_normal_randn_cpu(self):
        """Test that torch.randn with a seeded generator produces matching results."""
        gen = torch.manual_seed(42)
        y_spyre = torch.randn(3, 5, device="spyre", generator=gen)
        gen.manual_seed(42)
        y_cpu = torch.randn(3, 5, device="cpu", generator=gen)
        torch.testing.assert_close(y_spyre.to("cpu"), y_cpu, rtol=0.1, atol=0.1)

    def test_uniform_cpu(self):
        """Test that tensor.uniform_() produces values in [0, 1)."""
        x_spyre = torch.tensor(
            [[1, 2, 3], [4, 5, 6]], dtype=torch.float16, device="spyre"
        )
        x_spyre.uniform_()
        x_cpu = x_spyre.to("cpu")
        assert torch.all(x_cpu >= 0.0) and torch.all(x_cpu < 1.0), (
            f"uniform_ values out of range [0, 1): {x_cpu}"
        )
        assert not torch.all(x_cpu == x_cpu[0, 0]), (
            "uniform_ produced all identical values"
        )

    def test_uniform_custom_range_cpu(self):
        """Test that tensor.uniform_(-5, 5) produces values in [-5, 5)."""
        x_spyre = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float16, device="spyre"
        )
        x_spyre.uniform_(-5.0, 5.0)
        x_cpu = x_spyre.to("cpu")
        assert torch.all(x_cpu >= -5.0) and torch.all(x_cpu < 5.0), (
            f"uniform_ values out of range [-5, 5): {x_cpu}"
        )
        assert not torch.all(x_cpu == x_cpu[0]), (
            "uniform_ produced all identical values"
        )

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_tril_cpu(self, x):
        def fn(input):
            return torch.tril(input)

        compare_with_cpu(fn, x)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_triu_cpu(self, x, diagonal):
        def fn(input, diagonal):
            return torch.triu(input, diagonal)

        compare_with_cpu(fn, x, diagonal)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_sdpa_cpu(self, q, k, v, attn_mask, is_causal, enable_gqa):
        def fn(q, k, v, attn_mask, is_causal, enable_gqa):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask, is_causal=is_causal, enable_gqa=enable_gqa
            )

        compare_with_cpu(fn, q, k, v, attn_mask, is_causal, enable_gqa)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_implicit_loading(self):
        def test(end, device=None):
            return torch.arange(end, device=device, dtype=torch.float16)

        compiled = torch.compile(test, backend="inductor")
        output = compiled(64.0, device="spyre")

        _ = output.cpu()

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_item_cpu(self, *args):
        """Test .item() operation on Spyre tensors"""
        if len(args) == 1:
            x = args[0]

            def fn(t):
                return t.item()

            compare_with_cpu(fn, x, cpu_compile=False)

        elif len(args) == 2:
            x, y = args

            def fn(a, b):
                result = a * b
                return result.item()

            compare_with_cpu(fn, x, y, cpu_compile=False)

    def test_split_cpu(self, op, dim, index, x):
        def fn(x):
            return op(dim, index, x)

        compare_with_cpu(fn, x, run_eager=False)

    def test_slice_cpu(self, op, dim, index, x):
        def fn(x):
            start = index * (x.size()[0] // 3)
            end = (index + 1) * (x.size()[0] // 3)
            if dim == 0:
                return op(dim, index, x[start:end])
            elif dim == 1:
                return op(dim, index, x[:, start:end])
            elif dim == 2:
                return op(dim, index, x[:, :, start:end])

        compare_with_cpu(fn, x, run_eager=False, cpu_compile=False)

    def test_rope_cpu(self, q, freqs):
        def fn(q, freqs):
            B, S, E = q.shape
            D = freqs.shape[-1] * 2
            H = E // D
            q_ = q.view(B, S, H, D).view(B, S, H, 2, D // 2)
            mul_out = freqs[:, :, None, :, :, :] * q_.unsqueeze(-3)
            sum_out = mul_out.sum(4, keepdim=True)
            q_out = sum_out.flatten(3)
            return q_out

        compare_with_cpu(fn, q, freqs, cpu_compile=False)

    def test_attn_qkv_paths(self, q, k, v):
        # This tests the dataflows between rope/qkv projection and SDPA for q, k, and v
        def fn(q, k, v):
            B, Sq, Hq = q.shape[0:3]
            D = q.shape[-1] * 2
            Sk, Hk = k.shape[1:3]
            expansion = Hq // Hk
            # (post-rope) B S Hq 2 1 D/2 --(view)-> B S Hq D --(transpose)-> B Hq S D -> identity (contiguous)
            q_attn = q.view(B, Sq, Hq, D).transpose(1, 2).contiguous()
            # (post-rope) B S Hk 2 1 D/2 --(view)-> B S Hk D --(transpose)-> B Hk S D --(unsqueeze)-> B Hk 1 S D --(expand)-> B Hk 4 S D --(flatten)-> B 4Hk S D --(transpose)-> B 4Hk D S -> restickify
            k_attn = (
                k.view(B, Sk, Hk, D)
                .transpose(1, 2)
                .unsqueeze(2)
                .expand(-1, -1, expansion, -1, -1)
                .flatten(1, 2)
                .transpose(2, 3)
                .contiguous()
            )
            # (post-v proj) B S Hv*D --(view)-> B S Hv D --(transpose)-> B Hv S D --(unsqueeze)-> B Hv 1 S D --(expand)-> B Hv 4 S D --(flatten)-> B 4Hk S D -> identity (contiguous)
            v_attn = (
                v.view(B, Sk, Hk, D)
                .transpose(1, 2)
                .unsqueeze(2)
                .expand(-1, -1, expansion, -1, -1)
                .flatten(1, 2)
                .contiguous()
            )
            return q_attn, k_attn, v_attn

        # TODO(aviros): Add support for missing eager ops and debug remaining issues to match eager results
        compare_with_cpu(fn, q, k, v, cpu_compile=False, run_eager=False)


if __name__ == "__main__":
    unittest.main()
