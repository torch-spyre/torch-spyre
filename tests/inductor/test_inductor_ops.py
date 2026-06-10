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
    _compile_and_run,
    cached_randn,
    cached_xavier,
    compare_with_cpu,
    make_param_dict,
    unique_randn_along_dim,
    shapes2key,
)
import utils_inductor
from torch_spyre._inductor.dtype_ops import DtypeOpTable
from torch_spyre._inductor.constants import IDENTITY_OP

POINTWISE_UNARY_OPS_DICT = {
    "abs": torch.abs,
    "cos": torch.cos,
    "exp": torch.exp,
    "floor": torch.floor,
    "neg": torch.neg,
    "reciprocal": torch.reciprocal,
    "relu": torch.relu,
    "sign": torch.sign,
    "sin": torch.sin,
    "tanh": torch.tanh,
}

POINTWISE_UNARY_OPS_FP32_DICT = {
    "ceil": torch.ceil,
    "floor": torch.floor,
}

POINTWISE_BINARY_OPS_DICT = {
    "add": torch.add,
    "mul": torch.mul,
    "sub": torch.sub,
    "div": torch.div,
    "minimum": torch.minimum,
    "maximum": torch.maximum,
}

POINTWISE_BINARY_OPS_INT64_DICT = {
    "add": torch.add,
    "mul": torch.mul,
    "sub": torch.sub,
    "minimum": torch.minimum,
    "maximum": torch.maximum,
}

CORE_REDUCTION_OPS_DICT = {
    "sum": torch.sum,
    "mean": torch.mean,
    "amin": torch.amin,
    "amax": torch.amax,
}


COMMON_REDUCTION_KEEPDIM_PARAM_SETS = {
    # Regular single-dim coverage. Use moderate scale to reduce FP16 noise
    # without pushing values into the quantization floor.
    "2d_dim_0": (0, cached_randn((67, 256), scale=0.1)),
    "2d_dim_neg1": (-1, cached_randn((67, 256), scale=0.1)),
    "3d_dim_0": (0, cached_randn((3, 5, 256), scale=0.1)),
    "3d_dim_1": (1, cached_randn((67, 71, 256), scale=0.1)),
    "3d_dim_neg1": (-1, cached_randn((67, 71, 256), scale=0.1)),
    "4d_dim_0": (0, cached_randn((6, 7, 12, 256), scale=0.1)),
    "4d_dim_1": (1, cached_randn((6, 7, 12, 256), scale=0.1)),
    "4d_dim_2": (2, cached_randn((6, 7, 12, 256), scale=0.1)),
    "4d_dim_neg1": (-1, cached_randn((6, 7, 12, 256), scale=0.1)),
    "5d_dim_0": (0, cached_randn((2, 3, 5, 7, 256), scale=0.1)),
    "5d_dim_1": (1, cached_randn((2, 3, 5, 7, 256), scale=0.1)),
    "5d_dim_2": (2, cached_randn((2, 3, 5, 7, 256), scale=0.1)),
    "5d_dim_3": (3, cached_randn((2, 3, 5, 7, 256), scale=0.1)),
    "5d_dim_neg1": (-1, cached_randn((2, 3, 5, 7, 256), scale=0.1)),
    # SDSC padding-path coverage.
    "pad_2d_dim_0": (0, cached_randn((63, 129), scale=0.1)),
    "pad_2d_dim_1": (1, cached_randn((63, 129), scale=0.1)),
    "pad_3d_dim_0": (0, cached_randn((3, 7, 9), scale=0.1)),
    "pad_3d_dim_1": (1, cached_randn((3, 7, 9), scale=0.1)),
    # TODO: compiled mean(dim=2) on padded 3D tensors mismatches on spyre (issue #1706)
    # "pad_3d_dim_2": (2, cached_randn((3, 7, 9), scale=0.1)),
    "pad_4d_dim_0": (0, cached_randn((3, 7, 9, 32), scale=0.1)),
    "pad_4d_dim_1": (1, cached_randn((3, 7, 9, 32), scale=0.1)),
    "pad_4d_dim_2": (2, cached_randn((3, 7, 9, 32), scale=0.1)),
    "pad_4d_dim_3": (3, cached_randn((3, 7, 9, 32), scale=0.1)),
}


CORE_REDUCTION_EDGE_KEEPDIM_PARAM_SETS = {
    # TODO: empty tensors currently segfault during CPU->Spyre copy (issue #992)
    # "empty_2d_dim_0": (0, torch.empty((0, 256), dtype=torch.float16)),
    "large_2d_dim_0": (0, cached_randn((2048, 4096), scale=0.01)),
    "large_2d_dim_neg1": (-1, cached_randn((2048, 4096), scale=0.01)),
    "large_2d_4096_dim_0": (0, cached_randn((4096, 4096), scale=0.01)),
}


COMMON_REDUCTION_MULTIDIM_KEEPDIM_PARAM_SETS = {
    # Regular multidim coverage. Use lower scale to limit FP16 accumulation
    # noise across multiple reduced axes.
    "2d_dim_01_all": ((0, 1), cached_randn((67, 256), scale=0.01)),
    "3d_dim_01": ((0, 1), cached_randn((67, 71, 256), scale=0.01)),
    "3d_dim_02": ((0, 2), cached_randn((67, 71, 256), scale=0.01)),
    "3d_dim_12": ((1, 2), cached_randn((67, 71, 256), scale=0.01)),
    "3d_dim_012_all": ((0, 1, 2), cached_randn((67, 71, 256), scale=0.01)),
    "3d_neg_21": ((-2, -1), cached_randn((5, 7, 64), scale=0.01)),
    "3d_mixed_1_neg1": ((1, -1), cached_randn((5, 7, 64), scale=0.01)),
    "4d_dim_01": ((0, 1), cached_randn((6, 7, 12, 256), scale=0.01)),
    "4d_dim_02": ((0, 2), cached_randn((6, 7, 12, 256), scale=0.01)),
    "4d_dim_03": ((0, 3), cached_randn((6, 7, 12, 256), scale=0.01)),
    "4d_dim_12": ((1, 2), cached_randn((6, 7, 12, 256), scale=0.01)),
    "4d_dim_13": ((1, 3), cached_randn((6, 7, 12, 256), scale=0.01)),
    "4d_dim_23": ((2, 3), cached_randn((6, 7, 12, 64), scale=0.01)),
    "4d_dim_012": ((0, 1, 2), cached_randn((6, 7, 12, 256), scale=0.01)),
    "4d_dim_013": ((0, 1, 3), cached_randn((6, 7, 12, 256), scale=0.01)),
    "4d_dim_023": ((0, 2, 3), cached_randn((6, 7, 12, 256), scale=0.01)),
    "4d_dim_123": ((1, 2, 3), cached_randn((6, 7, 12, 256), scale=0.01)),
    "4d_dim_0123_all": ((0, 1, 2, 3), cached_randn((6, 7, 12, 64), scale=0.01)),
    "4d_unsorted_30": ((3, 0), cached_randn((4, 6, 8, 64), scale=0.01)),
    "4d_size1_23": ((2, 3), cached_randn((4, 6, 1, 64), scale=0.01)),
    "5d_dim_04": ((0, 4), cached_randn((2, 3, 5, 7, 256), scale=0.01)),
    "5d_dim_024": ((0, 2, 4), cached_randn((2, 3, 5, 7, 256), scale=0.01)),
    "5d_dim_1234": ((1, 2, 3, 4), cached_randn((2, 3, 5, 7, 256), scale=0.01)),
    "5d_mixed_1_neg1": ((1, -1), cached_randn((2, 3, 5, 7, 256), scale=0.01)),
    "5d_size1_34": ((3, 4), cached_randn((2, 3, 5, 1, 64), scale=0.01)),
    # SDSC padding-path coverage.
    "pad_2d_dim_01_all": ((0, 1), cached_randn((63, 129), scale=0.01)),
    "pad_3d_dim_01": ((0, 1), cached_randn((3, 7, 9), scale=0.01)),
    "pad_3d_dim_12": ((1, 2), cached_randn((3, 7, 9), scale=0.01)),
    "pad_3d_dim_012_all": ((0, 1, 2), cached_randn((3, 7, 9), scale=0.01)),
    "pad_4d_dim_23": ((2, 3), cached_randn((3, 7, 9, 32), scale=0.01)),
    "pad_4d_dim_0123_all": ((0, 1, 2, 3), cached_randn((3, 7, 9, 32), scale=0.01)),
    "pad_5d_dim_234": ((2, 3, 4), cached_randn((2, 3, 5, 7, 9), scale=0.01)),
}


CORE_REDUCTION_EDGE_MULTIDIM_KEEPDIM_PARAM_SETS = {
    # TODO: 5D all-dims sum/mean reduction is incorrect on spyre (issue #1707)
    # "5d_dim_01234_all": ((0, 1, 2, 3, 4), cached_randn((2, 3, 5, 7, 256), scale=0.1)),
    # "large_2d_dim_01_all": ((0, 1), cached_randn((2048, 4096), scale=0.01)),
    "large_3d_dim_12": ((1, 2), cached_randn((32, 64, 512), scale=0.01)),
}


INDEX_REDUCTION_KEEPDIM_PARAM_SETS = {
    name: (
        dim,
        unique_randn_along_dim(tuple(x.shape), dim=dim, dtype=x.dtype),
    )
    for name, (dim, x) in COMMON_REDUCTION_KEEPDIM_PARAM_SETS.items()
}


VECTOR_NORM_KEEPDIM_PARAM_SETS = {
    "ord1_2d_dim_0": (1, 0, cached_randn((67, 256))),
    "ord2_2d_dim_neg1": (2, -1, cached_randn((67, 256))),
    "ord2_3d_dim_12": (2, (1, 2), cached_randn((5, 7, 64))),
    "ord2_4d_size1_dim_2": (2, 2, cached_randn((4, 6, 1, 64))),
    "ordinf_4d_dim_neg1": (float("inf"), -1, cached_randn((6, 7, 12, 64))),
    "ordneginf_4d_dim_23": (
        -float("inf"),
        (2, 3),
        cached_randn((4, 6, 8, 64)),
    ),
    "ord2_5d_dim_1234": (2, (1, 2, 3, 4), cached_randn((2, 3, 5, 7, 64))),
    "ord2_5d_mixed_1_neg1": (2, (1, -1), cached_randn((2, 3, 5, 7, 64))),
    "ord1_pad_2d_dim_1": (1, 1, cached_randn((63, 129))),
    "ord2_pad_5d_dim_234": (2, (2, 3, 4), cached_randn((2, 3, 5, 7, 9))),
}


SPYRE_MODE_SUPPORT_OVERRIDES_BY_OP = {
    torch.amin: {
        "compiled": True,
        "eager": False,
        "reason": "Spyre eager aten::amin.out is not supported yet (issue #1708)",
    },
    torch.amax: {
        "compiled": True,
        "eager": False,
        "reason": "Spyre eager aten::amax.out is not supported yet (issue #1708)",
    },
    torch.min: {
        "compiled": True,
        "eager": False,
        "reason": "Spyre eager aten::min.dim_min is not supported yet",
    },
    torch.aminmax: {
        "compiled": True,
        "eager": False,
        "reason": "Spyre eager aten::aminmax.out is not supported yet",
    },
    torch.linalg.vector_norm: {
        "compiled": True,
        "eager": False,
        "reason": "Spyre eager linalg.vector_norm misroutes ord on Spyre",
    },
    torch.linalg.matrix_norm: {
        "compiled": True,
        "eager": False,
        "reason": "Spyre eager linalg.matrix_norm misroutes ord on Spyre",
    },
    torch.linalg.norm: {
        "compiled": True,
        "eager": False,
        "reason": "Spyre eager linalg.norm misroutes ord on Spyre",
    },
}


def _get_spyre_mode_support(op):
    return SPYRE_MODE_SUPPORT_OVERRIDES_BY_OP.get(
        op,
        {"compiled": True, "eager": True, "reason": None},
    )


def _compare_op_with_cpu(fn, op, *args, **kwargs):
    support = _get_spyre_mode_support(op)
    if not support["compiled"] and not support["eager"]:
        pytest.skip(support["reason"] or f"{op} is not supported on Spyre yet")
    kwargs.setdefault("cpu_compile", True)
    compare_with_cpu(
        fn,
        *args,
        run_compile=support["compiled"],
        run_eager=support["eager"],
        **kwargs,
    )


ALL_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.bool,
]

ALL_DTYPE_PAIRS = [(src, dst) for src in ALL_DTYPES for dst in ALL_DTYPES if src != dst]

TO_DTYPE_OP_SHAPES_UNALIGNED = [
    (68,),  # 1D unaligned: 68 > 1 fp16 stick (64 elems), not a multiple of 64
    (4, 16),
    (4, 68),
]

TO_DTYPE_OP_SHAPES_ALIGNED = [
    (64,),  # 1D aligned: exactly 1 fp16 stick — regression for 1D dtype-conv crash
    (5120,),  # 1D aligned: 80 fp16 sticks — exact repro shape from the bug report
    (4, 64),
    (4, 8, 128),
    (2, 4, 8, 64),
]

TO_DTYPE_OP_SHAPES = TO_DTYPE_OP_SHAPES_UNALIGNED + TO_DTYPE_OP_SHAPES_ALIGNED


def _dtype_name(dt):
    return str(dt).split(".")[-1]


TO_DTYPE_OP_MAP_PARAMS_SETS = {
    f"{_dtype_name(src)}_to_{_dtype_name(dst)}": (src, dst)
    for src, dst in ALL_DTYPE_PAIRS
}

TO_DTYPE_OP_PARAMS_SETS = {
    f"{_dtype_name(src)}_to_{_dtype_name(dst)}_{shapes2key((shape,))}": (
        cached_randn(shape, dtype=src),
        dst,
    )
    for src, dst in DtypeOpTable.get_dtype_pairs()
    for shape in TO_DTYPE_OP_SHAPES
    if src != torch.bool and dst != torch.bool
}


_DTYPE_OP_ALL_OPS_FAIL_SHAPES = {(4, 68), (68,)}

TO_DTYPE_OP_EXPECT_FAIL = [
    f"{_dtype_name(src)}_to_{_dtype_name(dst)}_{shapes2key((shape,))}"
    for src, dst in DtypeOpTable.get_dtype_pairs()
    for shape in TO_DTYPE_OP_SHAPES
    if (
        shape in _DTYPE_OP_ALL_OPS_FAIL_SHAPES
        or DtypeOpTable.get_operator(src, dst) != IDENTITY_OP
    )
]

TO_DTYPE_OP_ROUND_TRIP_PARAMS_SETS = {
    f"{_dtype_name(src)}_to_{_dtype_name(dst)}_{shapes2key((shape,))}": (
        cached_randn(shape, dtype=src),
        dst,
    )
    for src, dst in [(torch.float16, torch.float32)]
    for shape in TO_DTYPE_OP_SHAPES
}

TO_DTYPE_OP_ROUND_TRIP_EXPECT_FAIL = [
    f"{_dtype_name(src)}_to_{_dtype_name(dst)}_{shapes2key((shape,))}"
    for src, dst in [(torch.float16, torch.float32)]
    for shape in TO_DTYPE_OP_SHAPES_UNALIGNED
]

FP32_EPS = torch.finfo(torch.float32).eps  # 1.1920928955078125e-07
FP16_EPS = torch.finfo(torch.float16).eps  # 0.0009765625


class TestOps(unittest.TestCase, metaclass=ParameterizedTestMeta):
    torch.manual_seed(0xAFFE)  # seeds cached_randn/cached_xavier calls in PARAMS below

    def setUp(self):
        super().setUp()
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
            "test_sqrt_fp32",
            "test_unary_op",
        ): {
            "ops_dict": {
                "sqrt": torch.sqrt,  # undefined for negative input
            },
            "param_sets": {
                "1d_abs_fp32": (cached_randn((64,), abs=True, dtype=torch.float32),),
                "2d_abs_fp32": (
                    cached_randn((67, 256), abs=True, dtype=torch.float32),
                ),
                "3d_abs_fp32": (
                    cached_randn((32, 64, 128), abs=True, dtype=torch.float32),
                ),
            },
        },
        (
            "test_rsqrt_fp32",
            "test_unary_op",
        ): {
            "ops_dict": {
                "rsqrt": torch.rsqrt,  # undefined for zero or negative input
            },
            "param_sets": {
                "1d_abs_nz_fp32": (
                    cached_randn((64,), abs=True, dtype=torch.float32) + FP32_EPS,
                ),
                "2d_abs_nz_fp32": (
                    cached_randn((67, 256), abs=True, dtype=torch.float32) + FP32_EPS,
                ),
                "3d_abs_nz_fp32": (
                    cached_randn((32, 64, 128), abs=True, dtype=torch.float32)
                    + FP32_EPS,
                ),
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
        (
            "test_pointwise_binary_op_int64",
            "test_binary_op",
        ): {
            "ops_dict": POINTWISE_BINARY_OPS_INT64_DICT,
            "param_sets": {
                "1d": (
                    torch.randint(-100, 100, (256,), dtype=torch.int64),
                    torch.randint(-100, 100, (256,), dtype=torch.int64),
                ),
                "2d": (
                    torch.randint(-100, 100, (67, 256), dtype=torch.int64),
                    torch.randint(-100, 100, (67, 256), dtype=torch.int64),
                ),
                "3d": (
                    torch.randint(-100, 100, (67, 71, 256), dtype=torch.int64),
                    torch.randint(-100, 100, (67, 71, 256), dtype=torch.int64),
                ),
            },
        },
        ("test_add_broadcast", "test_add_broadcast"): {
            "param_sets": make_param_dict(
                [
                    ((256,), (67, 256)),
                ]
            ),
        },
        ("test_add_broadcast_cpu", "test_add_broadcast_cpu"): {
            "param_sets": make_param_dict(
                [
                    ((256,), (67, 256)),
                ]
            ),
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
            },
            "param_sets": make_param_dict(
                [
                    ((67, 256), (256, 128)),
                    # Padding
                    ((55, 2), (2, 99)),
                    ((67, 67), (67, 67)),
                    ((67, 255), (255, 128)),
                ],
                rand_type="xavier",
            ),
        },
        ("test_mm_autocast", "test_mm_autocast_cpu"): {
            "param_sets": {
                "fp32_enabled": (
                    True,
                    cached_randn((4, 4), dtype=torch.float32),
                    cached_randn((4, 4), differentiation=1, dtype=torch.float32),
                ),
                "f16_enabled": (
                    True,
                    cached_randn((4, 4), dtype=torch.float16),
                    cached_randn((4, 4), differentiation=1, dtype=torch.float16),
                ),
                "f16_disabled": (
                    False,
                    cached_randn((4, 4), differentiation=2, dtype=torch.float16),
                    cached_randn((4, 4), differentiation=3, dtype=torch.float16),
                ),
            },
            "expect_fail": ["fp32_enabled"],
        },
        ("test_einsum", "test_mm_relaxed"): {
            "ops_dict": {
                "einsum": lambda a, b: torch.einsum("mk, kn -> mn", a, b),
            },
            "param_sets": make_param_dict(
                [
                    ((67, 256), (256, 128)),
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
                    # Previous fail cases
                    # issue 502
                    ((32, 1, 2880), (32, 2880, 2880)),
                    # issue 1349
                    ((44, 1, 2880), (44, 2880, 2880)),
                    ((256, 1, 128), (256, 128, 512)),
                ],
                rand_type="xavier",
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
                ],
                rand_type="xavier",
            ),
        },
        ("test_matmul_noncontiguous", "test_mm_relaxed"): {
            "ops_dict": {"matmul": torch.matmul},
            "param_sets": {
                "3d": (
                    cached_xavier((128, 2, 128)).transpose(0, 1),
                    cached_xavier((128, 2, 256)).transpose(0, 1),
                ),
                "4d": (
                    cached_xavier((2, 8, 128, 128)),
                    cached_xavier((2, 128, 8, 128)).transpose(1, 2),
                ),
            },
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
        ("test_sub_scalar", "test_unary_op_cpu"): {
            "ops_dict": {
                "sub_scalar_5": lambda x: torch.sub(x, 5.0),
                "sub_scalar_neg": lambda x: torch.sub(x, -3.5),
                "sub_scalar_zero": lambda x: torch.sub(x, 0.0),
            },
            "param_sets": make_param_dict(
                [
                    ((256,),),
                    ((67, 256),),
                    ((67, 71, 256),),
                ]
            ),
        },
        ("test_sub_broadcast", "test_binary_op_cpu"): {
            "ops_dict": {"sub": torch.sub},
            "param_sets": {
                "1d_2d": (
                    cached_randn((256,)),
                    cached_randn((67, 256)),
                ),
                "2d_3d": (
                    cached_randn((71, 256)),
                    cached_randn((67, 71, 256)),
                ),
                "scalar_broadcast": (
                    cached_randn((1,)),
                    cached_randn((67, 256)),
                ),
                "3d_4d": (
                    cached_randn((12, 32, 64)),
                    cached_randn((7, 12, 32, 64)),
                ),
            },
        },
        ("test_sub_alpha", "test_binary_op_cpu"): {
            "ops_dict": {
                "sub_alpha_2": lambda a, b: torch.sub(a, b, alpha=2.0),
                "sub_alpha_0.5": lambda a, b: torch.sub(a, b, alpha=0.5),
                "sub_alpha_neg": lambda a, b: torch.sub(a, b, alpha=-1.0),
            },
            "param_sets": make_param_dict(
                [
                    ((256,),) * 2,
                    ((67, 256),) * 2,
                ]
            ),
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
        ("test_max_default", "test_reduce_cpu"): {
            "ops_dict": {
                "max": torch.max,
            },
            "param_sets": {
                "1d_float16": (unique_randn_along_dim((64,), dtype=torch.float16),),
                "2d_float16": (unique_randn_along_dim((8, 64), dtype=torch.float16),),
                "3d_float16": (
                    unique_randn_along_dim((2, 4, 64), dtype=torch.float16),
                ),
                "1d_int64": (unique_randn_along_dim((64,), dtype=torch.int64),),
                "2d_int64": (unique_randn_along_dim((67, 256), dtype=torch.int64),),
            },
            "expect_fail": [
                "1d_float16",
                "2d_float16",
                "3d_float16",
                "1d_int64",
                "2d_int64",
            ],
        },
        # Compare with cpu for now to avoid hitting eager mode coverage issue
        ("test_max_keepdim0", "test_reduce_keepdim0_cpu"): {
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
        ("test_max_keepdim1", "test_reduce_keepdim1_cpu"): {
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
        ("test_topk", "test_topk_cpu"): {
            "param_sets": {
                "2d_k4_dim0": (unique_randn_along_dim((64, 256), dim=0), 4, 0),
                "2d_k4_dim_minusone": (
                    unique_randn_along_dim((64, 256), dim=-1),
                    4,
                    -1,
                ),
                # "2d_k4_dim0_lessthanstick": (unique_randn_along_dim((8, 32), dim=0), 4, 0),
                # "2d_k4_dim_minusone_lessthanstick": (unique_randn_along_dim((1, 32), dim=-1), 4, -1),
            },
        },
        ("test_reduce_keepdim0", "test_reduce_keepdim0_cpu"): {
            "ops_dict": CORE_REDUCTION_OPS_DICT,
            "param_sets": COMMON_REDUCTION_KEEPDIM_PARAM_SETS,
        },
        ("test_reduce_keepdim1", "test_reduce_keepdim1_cpu"): {
            "ops_dict": CORE_REDUCTION_OPS_DICT,
            "param_sets": COMMON_REDUCTION_KEEPDIM_PARAM_SETS,
            "expect_fail": [
                "mean_fp16_3d_dim_2",
                "mean_fp16_3d_dim_neg1",
                "mean_fp32_3d_dim_2",
                "mean_fp32_3d_dim_neg1",
                "sum_fp32_3d_dim_neg1",
            ],
        },
        ("test_reduce_edge_keepdim0", "test_reduce_keepdim0_cpu"): {
            "ops_dict": CORE_REDUCTION_OPS_DICT,
            "param_sets": CORE_REDUCTION_EDGE_KEEPDIM_PARAM_SETS,
        },
        ("test_reduce_edge_keepdim1", "test_reduce_keepdim1_cpu"): {
            "ops_dict": CORE_REDUCTION_OPS_DICT,
            "param_sets": CORE_REDUCTION_EDGE_KEEPDIM_PARAM_SETS,
        },
        ("test_reduce_multidim_keepdim0", "test_reduce_multidim_keepdim0_cpu"): {
            "ops_dict": CORE_REDUCTION_OPS_DICT,
            "param_sets": COMMON_REDUCTION_MULTIDIM_KEEPDIM_PARAM_SETS,
        },
        ("test_reduce_multidim_keepdim1", "test_reduce_multidim_keepdim1_cpu"): {
            "ops_dict": CORE_REDUCTION_OPS_DICT,
            "param_sets": COMMON_REDUCTION_MULTIDIM_KEEPDIM_PARAM_SETS,
        },
        ("test_reduce_edge_multidim_keepdim0", "test_reduce_multidim_keepdim0_cpu"): {
            "ops_dict": CORE_REDUCTION_OPS_DICT,
            "param_sets": CORE_REDUCTION_EDGE_MULTIDIM_KEEPDIM_PARAM_SETS,
        },
        ("test_reduce_edge_multidim_keepdim1", "test_reduce_multidim_keepdim1_cpu"): {
            "ops_dict": CORE_REDUCTION_OPS_DICT,
            "param_sets": CORE_REDUCTION_EDGE_MULTIDIM_KEEPDIM_PARAM_SETS,
        },
        ("test_mean_layout_multidim_keepdim0", "test_reduce_multidim_keepdim0_cpu"): {
            "ops_dict": {
                "mean": torch.mean,
            },
            "param_sets": {
                "5d_permuted_dim_1_neg1": (
                    (1, -1),
                    cached_randn((2, 48, 2, 256, 65), scale=0.01).permute(
                        0, 2, 3, 4, 1
                    ),
                ),
            },
        },
        ("test_mean_layout_multidim_keepdim1", "test_reduce_multidim_keepdim1_cpu"): {
            "ops_dict": {
                "mean": torch.mean,
            },
            "param_sets": {
                "5d_permuted_dim_1_neg1": (
                    (1, -1),
                    cached_randn((2, 48, 2, 256, 65), scale=0.01).permute(
                        0, 2, 3, 4, 1
                    ),
                ),
            },
        },
        ("test_min_keepdim0", "test_reduce_keepdim0_cpu"): {
            "ops_dict": {
                "min": torch.min,
            },
            "param_sets": INDEX_REDUCTION_KEEPDIM_PARAM_SETS,
        },
        ("test_min_keepdim1", "test_reduce_keepdim1_cpu"): {
            "ops_dict": {
                "min": torch.min,
            },
            "param_sets": INDEX_REDUCTION_KEEPDIM_PARAM_SETS,
        },
        ("test_aminmax_keepdim0", "test_tuple_reduce_keepdim0_cpu"): {
            "ops_dict": {
                "aminmax": torch.aminmax,
            },
            "param_sets": INDEX_REDUCTION_KEEPDIM_PARAM_SETS,
        },
        ("test_aminmax_keepdim1", "test_tuple_reduce_keepdim1_cpu"): {
            "ops_dict": {
                "aminmax": torch.aminmax,
            },
            "param_sets": INDEX_REDUCTION_KEEPDIM_PARAM_SETS,
        },
        ("test_vector_norm_keepdim0", "test_norm_keepdim0_cpu"): {
            "ops_dict": {
                "vector_norm": torch.linalg.vector_norm,
            },
            "param_sets": VECTOR_NORM_KEEPDIM_PARAM_SETS,
        },
        ("test_vector_norm_keepdim1", "test_norm_keepdim1_cpu"): {
            "ops_dict": {
                "vector_norm": torch.linalg.vector_norm,
            },
            "param_sets": VECTOR_NORM_KEEPDIM_PARAM_SETS,
        },
        ("test_matrix_norm_keepdim0", "test_norm_keepdim0_cpu"): {
            "ops_dict": {
                "matrix_norm": torch.linalg.matrix_norm,
            },
            "param_sets": {
                "fro_3d_dim_12": ("fro", (1, 2), cached_randn((2, 3, 4))),
                "ord1_4d_dim_23": (1, (2, 3), cached_randn((2, 5, 7, 8))),
                "ordinf_5d_dim_34": (
                    float("inf"),
                    (3, 4),
                    cached_randn((2, 3, 5, 7, 8)),
                ),
            },
        },
        ("test_matrix_norm_keepdim1", "test_norm_keepdim1_cpu"): {
            "ops_dict": {
                "matrix_norm": torch.linalg.matrix_norm,
            },
            "param_sets": {
                "fro_3d_dim_12": ("fro", (1, 2), cached_randn((2, 3, 4))),
                "ord1_4d_dim_23": (1, (2, 3), cached_randn((2, 5, 7, 8))),
                "ordinf_5d_dim_34": (
                    float("inf"),
                    (3, 4),
                    cached_randn((2, 3, 5, 7, 8)),
                ),
            },
        },
        ("test_linalg_norm_keepdim0", "test_norm_keepdim0_cpu"): {
            "ops_dict": {
                "linalg_norm": torch.linalg.norm,
            },
            "param_sets": {
                "vector_2d_dim_1": (2, 1, cached_randn((67, 256))),
                "matrix_3d_dim_12": ("fro", (1, 2), cached_randn((2, 3, 4))),
                "matrix_4d_dim_23": ("fro", (2, 3), cached_randn((2, 5, 7, 8))),
            },
        },
        ("test_linalg_norm_keepdim1", "test_norm_keepdim1_cpu"): {
            "ops_dict": {
                "linalg_norm": torch.linalg.norm,
            },
            "param_sets": {
                "vector_2d_dim_1": (2, 1, cached_randn((67, 256))),
                "matrix_3d_dim_12": ("fro", (1, 2), cached_randn((2, 3, 4))),
                "matrix_4d_dim_23": ("fro", (2, 3), cached_randn((2, 5, 7, 8))),
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
                    ((49159, 4096),),
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
                    ((49159, 4096),),
                ]
            ),
            "expect_fail": ["49159x4096"],
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
                "large_dim_0_1_nopad": (
                    0,
                    1,
                    cached_randn((769, 4096, 63), abs=True),
                ),
                "large_dim_0_2_nopad": (
                    0,
                    2,
                    cached_randn((769, 4096, 63), abs=True),
                ),
                "large_dim_1_2_nopad": (
                    1,
                    2,
                    cached_randn((769, 4096, 63), abs=True),
                ),
                "large_dim_0_1": (
                    0,
                    1,
                    cached_randn((769, 4096, 64), abs=True),
                ),
                "large_dim_0_2": (
                    0,
                    2,
                    cached_randn((769, 4096, 64), abs=True),
                ),
                "large_dim_1_2": (
                    1,
                    2,
                    cached_randn((769, 4096, 64), abs=True),
                ),
            },
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
                "large_dim_0_1_nopad": (
                    0,
                    1,
                    cached_randn((769, 4096, 63), abs=True),
                ),
                "large_dim_0_2_nopad": (
                    0,
                    2,
                    cached_randn((769, 4096, 63), abs=True),
                ),
                "large_dim_1_2_nopad": (
                    1,
                    2,
                    cached_randn((769, 4096, 63), abs=True),
                ),
                "large_dim_0_1": (
                    0,
                    1,
                    cached_randn((769, 4096, 64), abs=True),
                ),
                "large_dim_0_2": (
                    0,
                    2,
                    cached_randn((769, 4096, 64), abs=True),
                ),
                "large_dim_1_2": (
                    1,
                    2,
                    cached_randn((769, 4096, 64), abs=True),
                ),
            },
            "expect_fail": [
                "large_dim_0_1",
                "large_dim_0_1_nopad",
                "large_dim_0_2",
                "large_dim_0_2_nopad",
                "large_dim_1_2",
                "large_dim_1_2_nopad",
            ],
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
                "signed_zero": (
                    torch.tensor([-0.0, 0.0, -0.0, 0.0], dtype=torch.float16),
                    torch.tensor([0.0, -0.0, 0.0, -0.0], dtype=torch.float16),
                ),
            },
        },
        ("test_cmp_scalar_int64", "test_cmp_scalar_int64_cpu"): {
            "ops_dict": {
                "ne": torch.ne,
            },
            "param_sets": {
                # [1, 64] int64 non-contiguous (stride (64,1)) != scalar
                "ne_1x64_int64_noncontig_eager": (
                    torch.randint(0, 100, (1, 64), dtype=torch.int64).as_strided(
                        (1, 64), (64, 1)
                    ),
                    0,
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
                "int32_1d": (torch.randint(0, 100, (128,), dtype=torch.int32),),
                "int32_2d": (torch.randint(0, 100, (256, 128), dtype=torch.int32),),
                "int32_3d": (torch.randint(0, 100, (8, 16, 26), dtype=torch.int32),),
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
        ("test_flatten", "test_flatten_cpu"): {
            "param_sets": {
                # 0D and 1D (identity cases)
                "0d_scalar": (0, -1, torch.tensor(42, dtype=torch.float16)),
                "1d_identity": (
                    0,
                    -1,
                    torch.tensor([10, 20, 30, 40, 50], dtype=torch.float16),
                ),
                # 2D tensors
                "2d_full": (
                    0,
                    -1,
                    torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float16),
                ),
                "2d_noop_dim0": (
                    0,
                    0,
                    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16),
                ),
                "2d_noop_dim1": (
                    1,
                    1,
                    torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16),
                ),
                # 3D tensors - contiguous
                "3d_full": (0, -1, cached_randn((2, 3, 4))),
                "3d_leading": (0, 1, cached_randn((2, 3, 4))),
                "3d_trailing": (1, 2, cached_randn((2, 3, 4))),
                # 4D tensors - contiguous
                "4d_full": (0, -1, cached_randn((2, 3, 4, 5))),
                "4d_middle": (1, 2, cached_randn((2, 3, 4, 5))),
                "4d_leading": (0, 2, cached_randn((2, 3, 4, 5))),
                "4d_trailing": (1, 3, cached_randn((2, 3, 4, 5))),
                # Negative dimensions
                "3d_neg_dims": (-2, -1, cached_randn((2, 3, 4))),
                "3d_neg_full": (-3, -1, cached_randn((2, 3, 4))),
                "3d_mixed_dims": (-3, 2, cached_randn((2, 3, 4))),
                # Non-contiguous tensors (after permute)
                "3d_noncontig_partial": (
                    1,
                    2,
                    torch.arange(24, dtype=torch.float16)
                    .reshape(2, 3, 4)
                    .permute(0, 2, 1),
                ),
                "3d_noncontig_full": (
                    0,
                    -1,
                    torch.arange(24, dtype=torch.float16)
                    .reshape(2, 3, 4)
                    .permute(2, 0, 1),
                ),
                # Edge cases
                "single_elem_1d": (0, -1, torch.ones((1,), dtype=torch.float16)),
                "single_elem_2d": (0, -1, torch.ones((1, 1), dtype=torch.float16)),
                "single_elem_3d": (0, -1, torch.ones((1, 1, 1), dtype=torch.float16)),
                # Large tensor
                "4d_large_middle": (1, 2, cached_randn((2, 8, 16, 32))),
                "4d_large_full": (0, -1, cached_randn((2, 8, 16, 32))),
            },
        },
        (
            "test_overwrite",
            "test_overwrite_cpu",
        ): {
            "param_sets": {
                "1d_dim0_single": (
                    cached_randn((64,), dtype=torch.float16),
                    cached_randn((256,), dtype=torch.float16),
                    [0],
                    [128],
                ),
                "1d_dim0_multi": (
                    cached_randn((128,), dtype=torch.float16),
                    cached_randn((256,), dtype=torch.float16),
                    [0],
                    [64],
                ),
                "2d_dim0_single": (
                    cached_randn((1, 256), dtype=torch.float16),
                    cached_randn((16, 256), dtype=torch.float16),
                    [0],
                    [8],
                ),
                "2d_dim0_multi": (
                    cached_randn((4, 256), dtype=torch.float16),
                    cached_randn((16, 256), dtype=torch.float16),
                    [0],
                    [3],
                ),
                "2d_dim1_single": (
                    cached_randn((8, 64), dtype=torch.float16),
                    cached_randn((8, 256), dtype=torch.float16),
                    [1],
                    [128],
                ),
                "2d_dim1_multi": (
                    cached_randn((8, 128), dtype=torch.float16),
                    cached_randn((8, 256), dtype=torch.float16),
                    [1],
                    [64],
                ),
                "3d_dim0_single": (
                    cached_randn((1, 4, 256), dtype=torch.float16),
                    cached_randn((8, 4, 256), dtype=torch.float16),
                    [0],
                    [3],
                ),
                "3d_dim0_multi": (
                    cached_randn((5, 4, 256), dtype=torch.float16),
                    cached_randn((8, 4, 256), dtype=torch.float16),
                    [0],
                    [2],
                ),
                "4d_dim0_single": (
                    cached_randn((1, 8, 4, 256), dtype=torch.float16),
                    cached_randn((4, 8, 4, 256), dtype=torch.float16),
                    [0],
                    [2],
                ),
                "4d_dim1_single": (
                    cached_randn((4, 1, 4, 256), dtype=torch.float16),
                    cached_randn((4, 8, 4, 256), dtype=torch.float16),
                    [1],
                    [3],
                ),
                "4d_dims01_multi": (
                    cached_randn((4, 3, 4, 128), dtype=torch.float16),
                    cached_randn((4, 8, 4, 256), dtype=torch.float16),
                    [1, 3],
                    [2, 128],
                ),
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
                "4d_dim2_zero": (
                    2,
                    cached_randn((0)),
                    cached_randn((1, 8, 14, 64), dtype=torch.float16),
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
                "4d_dim_m2_empty_first": (
                    -2,
                    torch.zeros(0, dtype=torch.float16),
                    cached_randn((1, 8, 14, 64), dtype=torch.float16),
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
                "2d_last_dim_negative_right": (
                    cached_randn((3, 256), dtype=torch.float16),
                    (0, -64),
                ),
                "2d_last_dim_negative_left": (
                    cached_randn((5, 256), dtype=torch.float16),
                    (-64, 0),
                ),
                "2d_last_dim_negative_both": (
                    cached_randn((5, 256), dtype=torch.float16),
                    (-64, -64),
                ),
                "2d_last_dim_mixed": (
                    cached_randn((5, 256), dtype=torch.float16),
                    (-64, 64),
                ),
                "2d_dim0_negative": (
                    cached_randn((5, 256), dtype=torch.float16),
                    (0, 0, -2, 0),
                ),
                "2d_dim0_mixed": (
                    cached_randn((5, 64), dtype=torch.float16),
                    (0, 0, -1, 2),
                ),
                "3d_last_dim_negative_right": (
                    cached_randn((2, 5, 128), dtype=torch.float16),
                    (0, -64),
                ),
                "3d_last_dim_mixed": (
                    cached_randn((2, 5, 128), dtype=torch.float16),
                    (64, -32),
                ),
                "3d_dim1_negative": (
                    cached_randn((2, 5, 128), dtype=torch.float16),
                    (0, 0, -2, 0),
                ),
                "3d_dim1_negative_both": (
                    cached_randn((2, 5, 128), dtype=torch.float16),
                    (0, 0, -2, -2),
                ),
                "3d_dim1_mixed": (
                    cached_randn((2, 5, 128), dtype=torch.float16),
                    (0, 0, -1, 2),
                ),
                "4d_dim1_mixed": (
                    cached_randn((2, 5, 8, 64), dtype=torch.float16),
                    (0, 0, 0, 0, -1, 2),
                ),
                "4d_dim2_negative_both": (
                    cached_randn((2, 5, 8, 64), dtype=torch.float16),
                    (0, 0, -2, -2),
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
            "expect_fail": ["value_2"],
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
                "fp16_signed_0": (
                    torch.tensor([0.0, -0.0, 1.0, -1.0], dtype=torch.float16),
                ),
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
                "float2bool": (
                    torch.zeros(128, dtype=torch.bool),  # bool tensor
                    (cached_randn((128,)) > 0).to(dtype=torch.float16),  # float tensor
                ),
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
                "3d2": (2, cached_randn((3, 4, 128))),
                "2d1": (1, cached_randn((4, 128))),
                "4d3": (3, cached_randn((2, 3, 4, 128))),
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
                "1d1": (1, cached_randn((4, 128)), cached_randn((4,))),
                "2d2": (2, cached_randn((3, 4, 128)), cached_randn((3, 4))),
                "3d3": (3, cached_randn((2, 3, 4, 128)), cached_randn((2, 3, 4))),
            },
            "expect_fail": ["1d1", "2d2", "3d3"],
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
                "2d_transposed": (
                    cached_randn((128, 256), dtype=torch.float16).transpose(0, 1),
                    cached_randn((128), dtype=torch.float16),
                    torch.zeros([128], dtype=torch.float16),
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
        ("test_mean_default", "test_mean_default_cpu"): {
            "param_sets": {
                "1d": (cached_randn((512,)),),
                "2d": (cached_randn((32, 64)),),
                "3d": (cached_randn((1, 11, 4096)),),
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
                "mha_decode": (
                    cached_randn(
                        (2, 1, 32, 128), differentiation=1, dtype=torch.float16
                    ),
                    cached_randn(
                        (2, 257, 32, 128), differentiation=2, dtype=torch.float16
                    ),
                    cached_randn(
                        (2, 257, 32, 128), differentiation=3, dtype=torch.float16
                    ),
                    False,
                    False,
                ),
                "gqa_decode": (
                    cached_randn(
                        (2, 1, 32, 128), differentiation=1, dtype=torch.float16
                    ),
                    cached_randn(
                        (2, 257, 8, 128), differentiation=2, dtype=torch.float16
                    ),
                    cached_randn(
                        (2, 257, 8, 128), differentiation=3, dtype=torch.float16
                    ),
                    False,
                    True,
                ),
            },
            "expect_fail": ["mha_decode", "gqa_decode"],
        },
        ("test_split", "test_split_cpu"): {
            "ops_dict": {
                "exp": (
                    lambda dim, index, x: (
                        torch.exp(torch.split(x, x.size()[dim] // 3, dim=dim)[index]),
                    )
                ),
                "add": (
                    lambda dim, index, x: (
                        y := torch.split(x, x.size()[dim] // 3, dim=dim),
                        index2 := (index + 1) % 3,
                        torch.add(y[index], y[index2]),
                    )[-1]
                ),
                "sum": (
                    lambda dim, index, x: (
                        torch.sum(
                            torch.split(x, x.size()[dim] // 3, dim=dim)[index],
                            dim=dim,
                            keepdim=True,
                        ),
                    )
                ),
                "amax": (
                    lambda dim, index, x: (
                        torch.amax(
                            torch.split(x, x.size()[dim] // 3, dim=dim)[index],
                            dim=dim,
                            keepdim=False,
                        ),
                    )
                ),
                "copy_": (
                    lambda dim, index, x: (
                        y := torch.split(x, x.size()[dim] // 3, dim=dim)[index],
                        y.copy_(torch.ones_like(y))._base,
                    )[-1]
                ),
                "chunk": (lambda dim, index, x: x.chunk(3, dim=dim)[index].clone()),
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
                "exp": lambda dim, x: torch.exp(x),
                "add": lambda dim, x: torch.add(x.clone(), x),
                "sum": lambda dim, x: torch.sum(x, dim=dim, keepdim=True),
                "amax": lambda dim, x: torch.amax(x, dim=dim, keepdim=False),
                "copy_": lambda dim, x: x.copy_(torch.ones_like(x))._base,
            },
            "param_sets": {
                "1d0s0": (0, 0, cached_randn((192,), dtype=torch.float16)),
                "1d0s1": (0, 1, cached_randn((192,), dtype=torch.float16)),
                "1d0s2": (0, 2, cached_randn((192,), dtype=torch.float16)),
                "2d0s0": (0, 0, cached_randn((3, 192), dtype=torch.float16)),
                "2d0s1": (0, 1, cached_randn((3, 192), dtype=torch.float16)),
                "2d0s2": (0, 2, cached_randn((3, 192), dtype=torch.float16)),
                "2d1s0": (1, 0, cached_randn((3, 192), dtype=torch.float16)),
                "2d1s1": (1, 1, cached_randn((3, 192), dtype=torch.float16)),
                "2d1s2": (1, 2, cached_randn((3, 192), dtype=torch.float16)),
                "3d0s0": (0, 0, cached_randn((3, 5, 192), dtype=torch.float16)),
                "3d0s1": (0, 1, cached_randn((3, 5, 192), dtype=torch.float16)),
                "3d0s2": (0, 2, cached_randn((3, 5, 192), dtype=torch.float16)),
                "3d1s0": (1, 0, cached_randn((5, 3, 192), dtype=torch.float16)),
                "3d1s1": (1, 1, cached_randn((5, 3, 192), dtype=torch.float16)),
                "3d1s2": (1, 2, cached_randn((5, 3, 192), dtype=torch.float16)),
                "3d2s0": (2, 0, cached_randn((3, 3, 192), dtype=torch.float16)),
                "3d2s1": (2, 1, cached_randn((3, 3, 192), dtype=torch.float16)),
                "3d2s2": (2, 2, cached_randn((3, 3, 192), dtype=torch.float16)),
            },
        },
        ("test_slice_synthetic_dims", "test_slice_synthetic_dims_cpu"): {
            "param_sets": {
                "5d": (cached_randn((2, 3, 4, 5, 192), dtype=torch.float16),),
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
        ("test_sum_keepdim1", "test_sum_eager"): {
            "ops_dict": {"sum": torch.sum},
            "param_sets": {
                "fp16_1d_dim_0": (0, True, cached_randn((64,), dtype=torch.float16)),
                "fp16_2d_dim_0": (
                    0,
                    True,
                    cached_randn((67, 256), dtype=torch.float16),
                ),
                "fp16_2d_dim_1": (
                    1,
                    True,
                    cached_randn((67, 256), dtype=torch.float16),
                ),
                "fp16_3d_dim_0": (
                    0,
                    True,
                    cached_randn((3, 5, 256), dtype=torch.float16, scale=0.1),
                ),
                "fp16_3d_dim_1": (
                    1,
                    True,
                    cached_randn((67, 71, 256), dtype=torch.float16, scale=0.1),
                ),
                "fp16_3d_dim_2": (
                    2,
                    True,
                    cached_randn((67, 71, 256), dtype=torch.float16, scale=0.1),
                ),
                "fp16_4d_dim_0": (
                    0,
                    True,
                    cached_randn((6, 7, 12, 256), dtype=torch.float16, scale=0.1),
                ),
                "fp16_4d_dim_1": (
                    1,
                    True,
                    cached_randn((6, 7, 12, 256), dtype=torch.float16, scale=0.1),
                ),
                "fp16_4d_dim_2": (
                    2,
                    True,
                    cached_randn((6, 7, 12, 256), dtype=torch.float16, scale=0.1),
                ),
                "fp16_4d_dim_3": (
                    3,
                    True,
                    cached_randn((6, 7, 12, 256), dtype=torch.float16, scale=0.1),
                ),
                "fp16_3d_dim_neg1": (
                    -1,
                    True,
                    cached_randn((3, 7, 9), dtype=torch.float16, scale=0.1),
                ),
                "fp16_3d_dim_neg2": (
                    -2,
                    True,
                    cached_randn((3, 7, 9), dtype=torch.float16, scale=0.1),
                ),
                "fp32_1d_dim_0": (0, True, cached_randn((64,), dtype=torch.float32)),
                "fp32_2d_dim_0": (
                    0,
                    True,
                    cached_randn((67, 256), dtype=torch.float32),
                ),
                "fp32_2d_dim_1": (
                    1,
                    True,
                    cached_randn((67, 256), dtype=torch.float32),
                ),
                "fp32_3d_dim_0": (
                    0,
                    True,
                    cached_randn((3, 5, 256), dtype=torch.float32, scale=0.1),
                ),
                "fp32_3d_dim_1": (
                    1,
                    True,
                    cached_randn((67, 71, 256), dtype=torch.float32, scale=0.1),
                ),
                "fp32_3d_dim_2": (
                    2,
                    True,
                    cached_randn((67, 71, 256), dtype=torch.float32, scale=0.1),
                ),
                "fp32_4d_dim_0": (
                    0,
                    True,
                    cached_randn((6, 7, 12, 256), dtype=torch.float32, scale=0.1),
                ),
                "fp32_4d_dim_1": (
                    1,
                    True,
                    cached_randn((6, 7, 12, 256), dtype=torch.float32, scale=0.1),
                ),
                "fp32_4d_dim_2": (
                    2,
                    True,
                    cached_randn((6, 7, 12, 256), dtype=torch.float32, scale=0.1),
                ),
                "fp32_4d_dim_3": (
                    3,
                    True,
                    cached_randn((6, 7, 12, 256), dtype=torch.float32, scale=0.1),
                ),
                "fp32_3d_dim_neg1": (
                    -1,
                    True,
                    cached_randn((3, 7, 9), dtype=torch.float32, scale=0.1),
                ),
                "fp32_3d_dim_neg2": (
                    -2,
                    True,
                    cached_randn((3, 7, 9), dtype=torch.float32, scale=0.1),
                ),
            },
        },
        ("test_sum_keepdim0", "test_sum_eager"): {
            "ops_dict": {"sum": torch.sum},
            "param_sets": {
                "fp16_2d_dim_0": (
                    0,
                    False,
                    cached_randn((67, 256), dtype=torch.float16),
                ),
                "fp16_2d_dim_1": (
                    1,
                    False,
                    cached_randn((67, 256), dtype=torch.float16),
                ),
                "fp16_3d_dim_1": (
                    1,
                    False,
                    cached_randn((67, 71, 256), dtype=torch.float16, scale=0.01),
                ),
                "fp16_3d_dim_2": (
                    2,
                    False,
                    cached_randn((67, 71, 256), dtype=torch.float16, scale=0.01),
                ),
                "fp16_4d_dim_0": (
                    0,
                    False,
                    cached_randn((6, 7, 12, 64), dtype=torch.float16, scale=0.01),
                ),
                "fp16_4d_dim_1": (
                    1,
                    False,
                    cached_randn((6, 7, 12, 64), dtype=torch.float16, scale=0.01),
                ),
                "fp16_4d_dim_2": (
                    2,
                    False,
                    cached_randn((6, 7, 12, 64), dtype=torch.float16, scale=0.01),
                ),
                "fp16_4d_dim_3": (
                    3,
                    False,
                    cached_randn((6, 7, 12, 64), dtype=torch.float16, scale=0.01),
                ),
                "fp32_2d_dim_0": (
                    0,
                    False,
                    cached_randn((67, 256), dtype=torch.float32),
                ),
                "fp32_2d_dim_1": (
                    1,
                    False,
                    cached_randn((67, 256), dtype=torch.float32),
                ),
                "fp32_3d_dim_1": (
                    1,
                    False,
                    cached_randn((67, 71, 256), dtype=torch.float32, scale=0.01),
                ),
                "fp32_3d_dim_2": (
                    2,
                    False,
                    cached_randn((67, 71, 256), dtype=torch.float32, scale=0.01),
                ),
                "fp32_4d_dim_0": (
                    0,
                    False,
                    cached_randn((6, 7, 12, 64), dtype=torch.float32, scale=0.01),
                ),
                "fp32_4d_dim_1": (
                    1,
                    False,
                    cached_randn((6, 7, 12, 64), dtype=torch.float32, scale=0.01),
                ),
                "fp32_4d_dim_2": (
                    2,
                    False,
                    cached_randn((6, 7, 12, 64), dtype=torch.float32, scale=0.01),
                ),
                "fp32_4d_dim_3": (
                    3,
                    False,
                    cached_randn((6, 7, 12, 64), dtype=torch.float32, scale=0.01),
                ),
            },
        },
        ("test_mean_keepdim1", "test_mean_eager"): {
            "ops_dict": {"mean": torch.mean},
            "param_sets": {
                "fp16_2d_dim_0": (
                    0,
                    True,
                    cached_randn((67, 256), dtype=torch.float16),
                ),
                "fp16_2d_dim_1": (
                    1,
                    True,
                    cached_randn((67, 256), dtype=torch.float16),
                ),
                "fp16_3d_dim_0": (
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
                "fp16_3d_dim_1": (
                    1,
                    True,
                    torch.tensor(
                        [
                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                        ],
                        dtype=torch.float16,
                    ),
                ),
                "fp16_3d_dim_2": (
                    2,
                    True,
                    torch.tensor(
                        [
                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                        ],
                        dtype=torch.float16,
                    ),
                ),
                "fp16_3d_dim_neg1": (
                    -1,
                    True,
                    cached_randn((3, 7, 9), dtype=torch.float16),
                ),
                "fp32_2d_dim_0": (
                    0,
                    True,
                    cached_randn((67, 256), dtype=torch.float32),
                ),
                "fp32_2d_dim_1": (
                    1,
                    True,
                    cached_randn((67, 256), dtype=torch.float32),
                ),
                "fp32_3d_dim_0": (
                    0,
                    True,
                    torch.tensor(
                        [
                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                        ],
                        dtype=torch.float32,
                    ),
                ),
                "fp32_3d_dim_1": (
                    1,
                    True,
                    torch.tensor(
                        [
                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                        ],
                        dtype=torch.float32,
                    ),
                ),
                "fp32_3d_dim_2": (
                    2,
                    True,
                    torch.tensor(
                        [
                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                        ],
                        dtype=torch.float32,
                    ),
                ),
                "fp32_3d_dim_neg1": (
                    -1,
                    True,
                    cached_randn((3, 7, 9), dtype=torch.float32),
                ),
            },
        },
        ("test_mean_keepdim0", "test_mean_eager"): {
            "ops_dict": {"mean": torch.mean},
            "param_sets": {
                "fp16_3d_dim_0": (
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
                "fp16_3d_dim_1": (
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
                "fp16_2d_dim_0": (
                    0,
                    False,
                    cached_randn((67, 256), dtype=torch.float16),
                ),
                "fp16_2d_dim_1": (
                    1,
                    False,
                    cached_randn((67, 256), dtype=torch.float16),
                ),
                "fp32_3d_dim_0": (
                    0,
                    False,
                    torch.tensor(
                        [
                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                        ],
                        dtype=torch.float32,
                    ),
                ),
                "fp32_3d_dim_1": (
                    1,
                    False,
                    torch.tensor(
                        [
                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                        ],
                        dtype=torch.float32,
                    ),
                ),
                "fp32_2d_dim_0": (
                    0,
                    False,
                    cached_randn((67, 256), dtype=torch.float32),
                ),
                "fp32_2d_dim_1": (
                    1,
                    False,
                    cached_randn((67, 256), dtype=torch.float32),
                ),
            },
            "expect_fail": [
                "fp16_3d_dim_2",
                "fp16_3d_dim_neg1",
                "fp32_2d_dim_0",
                "fp32_2d_dim_1",
                "fp32_3d_dim_0",
                "fp32_3d_dim_1",
                "fp32_3d_dim_2",
                "fp32_3d_dim_neg1",
            ],
        },
        ("test_max_keepdim1", "test_max_eager"): {
            "ops_dict": {"max": torch.max},
            "param_sets": {
                "fp16_2d_dim_0": (
                    0,
                    True,
                    unique_randn_along_dim(
                        (67, 256), dim=0, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_2d_dim_1": (
                    1,
                    True,
                    unique_randn_along_dim(
                        (67, 256), dim=1, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_3d_dim_0": (
                    0,
                    True,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=0, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_3d_dim_1": (
                    1,
                    True,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=1, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_3d_dim_2": (
                    2,
                    True,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=2, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_0": (
                    0,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=0, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_1": (
                    1,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=1, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_2": (
                    2,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=2, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_3": (
                    3,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=3, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp32_2d_dim_0": (
                    0,
                    True,
                    unique_randn_along_dim(
                        (67, 256), dim=0, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_2d_dim_1": (
                    1,
                    True,
                    unique_randn_along_dim(
                        (67, 256), dim=1, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_3d_dim_0": (
                    0,
                    True,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=0, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_3d_dim_1": (
                    1,
                    True,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=1, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_3d_dim_2": (
                    2,
                    True,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=2, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_0": (
                    0,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=0, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_1": (
                    1,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=1, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_2": (
                    2,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=2, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_3": (
                    3,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=3, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
            },
        },
        ("test_max_keepdim0", "test_max_eager"): {
            "ops_dict": {"max": torch.max},
            "param_sets": {
                "fp16_2d_dim_0": (
                    0,
                    False,
                    unique_randn_along_dim(
                        (67, 256), dim=0, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_2d_dim_1": (
                    1,
                    False,
                    unique_randn_along_dim(
                        (67, 256), dim=1, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_3d_dim_1": (
                    1,
                    False,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=1, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_3d_dim_2": (
                    2,
                    False,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=2, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_0": (
                    0,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=0, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_1": (
                    1,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=1, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_2": (
                    2,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=2, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_3": (
                    3,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=3, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp32_2d_dim_0": (
                    0,
                    False,
                    unique_randn_along_dim(
                        (67, 256), dim=0, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_2d_dim_1": (
                    1,
                    False,
                    unique_randn_along_dim(
                        (67, 256), dim=1, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_3d_dim_1": (
                    1,
                    False,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=1, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_3d_dim_2": (
                    2,
                    False,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=2, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_0": (
                    0,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=0, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_1": (
                    1,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=1, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_2": (
                    2,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=2, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_3": (
                    3,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=3, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
            },
        },
        ("test_min_keepdim1", "test_min_eager"): {
            "ops_dict": {"min": torch.min},
            "param_sets": {
                "fp16_2d_dim_0": (
                    0,
                    True,
                    unique_randn_along_dim(
                        (67, 256), dim=0, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_2d_dim_1": (
                    1,
                    True,
                    unique_randn_along_dim(
                        (67, 256), dim=1, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_3d_dim_0": (
                    0,
                    True,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=0, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_3d_dim_1": (
                    1,
                    True,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=1, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_3d_dim_2": (
                    2,
                    True,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=2, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_0": (
                    0,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=0, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_1": (
                    1,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=1, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_2": (
                    2,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=2, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_3": (
                    3,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=3, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp32_2d_dim_0": (
                    0,
                    True,
                    unique_randn_along_dim(
                        (67, 256), dim=0, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_2d_dim_1": (
                    1,
                    True,
                    unique_randn_along_dim(
                        (67, 256), dim=1, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_3d_dim_0": (
                    0,
                    True,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=0, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_3d_dim_1": (
                    1,
                    True,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=1, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_3d_dim_2": (
                    2,
                    True,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=2, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_0": (
                    0,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=0, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_1": (
                    1,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=1, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_2": (
                    2,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=2, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_3": (
                    3,
                    True,
                    unique_randn_along_dim(
                        (6, 7, 12, 256), dim=3, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
            },
        },
        ("test_min_keepdim0", "test_min_eager"): {
            "ops_dict": {"min": torch.min},
            "param_sets": {
                "fp16_2d_dim_0": (
                    0,
                    False,
                    unique_randn_along_dim(
                        (67, 256), dim=0, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_2d_dim_1": (
                    1,
                    False,
                    unique_randn_along_dim(
                        (67, 256), dim=1, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_3d_dim_1": (
                    1,
                    False,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=1, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_3d_dim_2": (
                    2,
                    False,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=2, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_0": (
                    0,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=0, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_1": (
                    1,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=1, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_2": (
                    2,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=2, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp16_4d_dim_3": (
                    3,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=3, dtype=torch.float16, seed=0xAFFE
                    ),
                ),
                "fp32_2d_dim_0": (
                    0,
                    False,
                    unique_randn_along_dim(
                        (67, 256), dim=0, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_2d_dim_1": (
                    1,
                    False,
                    unique_randn_along_dim(
                        (67, 256), dim=1, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_3d_dim_1": (
                    1,
                    False,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=1, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_3d_dim_2": (
                    2,
                    False,
                    unique_randn_along_dim(
                        (67, 71, 256), dim=2, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_0": (
                    0,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=0, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_1": (
                    1,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=1, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_2": (
                    2,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=2, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
                "fp32_4d_dim_3": (
                    3,
                    False,
                    unique_randn_along_dim(
                        (6, 17, 7, 64), dim=3, dtype=torch.float32, seed=0xAFFE
                    ),
                ),
            },
        },
        (
            "test_pointwise_unary_op_fp32",
            "test_unary_op",
        ): {
            "ops_dict": POINTWISE_UNARY_OPS_FP32_DICT,
            "param_sets": {
                "256": (cached_randn((256,), dtype=torch.float32),),
                "67x256": (cached_randn((67, 256), dtype=torch.float32),),
                "67x71x256": (cached_randn((67, 71, 256), dtype=torch.float32),),
            },
        },
        ("test_eq_scalar", "test_scalar_comparison_base"): {
            "param_sets": {
                "int_42": (
                    42,
                    torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float16),
                ),
                "int_10": (
                    10,
                    torch.tensor([1.0, 10.0, 5.0, 10.0, 3.0], dtype=torch.float16),
                ),
                "float_3_5": (
                    3.5,
                    torch.tensor([1.0, 3.5, 2.5, 3.1, 5.0], dtype=torch.float16),
                ),
                "negative_5": (
                    -5,
                    torch.tensor([-5, 0.0, 5.0, -5.0, 10.0], dtype=torch.float16),
                ),
                "zero": (
                    0,
                    torch.tensor([0.0, 1.0, -1.0, 0.0, 5.0], dtype=torch.float16),
                ),
            },
        },
        ("test_eq_scalar_multidim", "test_scalar_multidim_base"): {
            "param_sets": {
                "2d": (
                    7,
                    torch.tensor(
                        [[1.0, 7.0, 3.0], [7.0, 5.0, 7.0]], dtype=torch.float16
                    ),
                ),
                "3d": (7, torch.randint(0, 15, (3, 4, 5)).to(torch.float16)),
                "4d": (7, torch.randint(0, 15, (2, 3, 4, 5)).to(torch.float16)),
                "large": (42, torch.randn(100, 50, dtype=torch.float16)),
            },
        },
        ("test_eq_scalar_vs_tensor", "test_scalar_vs_tensor_base"): {
            "param_sets": {
                "mixed": (
                    torch.tensor([1.0, 5.0, 3.0, 5.0], dtype=torch.float16),
                    torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16),
                    5,
                ),
            },
        },
        ("test_where_default", "test_where_eager_default_fallback"): {
            "ops_dict": {"where": torch.where},
            "param_sets": {
                "fp16_2d": (cached_randn((10, 10), dtype=torch.float16) > 1,),
                "fp16_3d": (cached_randn((5, 10, 10), dtype=torch.float16) > 1,),
            },
        },
        ("test_where_self", "test_where_eager"): {
            "ops_dict": {"where": torch.where},
            "param_sets": {
                "fp16_2d": (
                    cached_randn((10, 10), dtype=torch.float16) > 1,
                    cached_randn((10, 10), dtype=torch.float16),
                    cached_randn((10, 10), dtype=torch.float16),
                ),
                "fp16_3d": (
                    cached_randn((5, 10, 10), dtype=torch.float16) > 1,
                    cached_randn((5, 10, 10), dtype=torch.float16),
                    cached_randn((5, 10, 10), dtype=torch.float16),
                ),
                "fp16_broadcast": (
                    cached_randn((10,), dtype=torch.float16) > 1,
                    cached_randn((5, 10), dtype=torch.float16),
                    cached_randn((5, 10), dtype=torch.float16),
                ),
            },
        },
        ("test_where_scalarother", "test_where_eager"): {
            "ops_dict": {"where": torch.where},
            "param_sets": {
                "fp16_2d": (
                    cached_randn((10, 10), dtype=torch.float16) > 1,
                    cached_randn((10, 10), dtype=torch.float16),
                    0,
                ),
                "fp16_3d": (
                    cached_randn((5, 10, 10), dtype=torch.float16) > 1,
                    cached_randn((5, 10, 10), dtype=torch.float16),
                    0,
                ),
                "fp16_broadcast": (
                    cached_randn((10,), dtype=torch.float16) > 1,
                    cached_randn((5, 10), dtype=torch.float16),
                    0,
                ),
            },
        },
        ("test_where_scalarself", "test_where_eager"): {
            "ops_dict": {"where": torch.where},
            "param_sets": {
                "fp16_2d": (
                    cached_randn((10, 10), dtype=torch.float16) > 1,
                    0,
                    cached_randn((10, 10), dtype=torch.float16),
                ),
                "fp16_3d": (
                    cached_randn((5, 10, 10), dtype=torch.float16) > 1,
                    0,
                    cached_randn((5, 10, 10), dtype=torch.float16),
                ),
                "fp16_broadcast": (
                    cached_randn((10,), dtype=torch.float16) > 1,
                    0,
                    cached_randn((5, 10), dtype=torch.float16),
                ),
            },
        },
        ("test_where_scalar", "test_where_eager_scalar"): {
            "ops_dict": {"where": torch.where},
            "param_sets": {
                "fp16_2d": (
                    cached_randn((10, 10), dtype=torch.float16) > 1,
                    0,
                    0,
                ),
                "fp16_3d": (
                    cached_randn((5, 10, 10), dtype=torch.float16) > 1,
                    0,
                    0,
                ),
                "fp16_broadcast": (
                    cached_randn((10,), dtype=torch.float16) > 1,
                    0,
                    0,
                ),
            },
        },
        ("test_where_self_out", "test_where_eager_selfout"): {
            "ops_dict": {"where": torch.where},
            "param_sets": {
                "fp16_2d": (
                    cached_randn((10, 10), dtype=torch.float16) > 1,
                    cached_randn((10, 10), dtype=torch.float16),
                    cached_randn((10, 10), dtype=torch.float16),
                    cached_randn((10, 10), dtype=torch.float16),
                ),
                "fp16_3d": (
                    cached_randn((5, 10, 10), dtype=torch.float16) > 1,
                    cached_randn((5, 10, 10), dtype=torch.float16),
                    cached_randn((5, 10, 10), dtype=torch.float16),
                    cached_randn((5, 10, 10), dtype=torch.float16),
                ),
                "fp16_broadcast": (
                    cached_randn((10,), dtype=torch.float16) > 1,
                    cached_randn((5, 10), dtype=torch.float16),
                    cached_randn((5, 10), dtype=torch.float16),
                    cached_randn((5, 10), dtype=torch.float16),
                ),
            },
        },
        ("test_to_dtype_op_map", "test_to_dtype_op_map"): {
            "param_sets": TO_DTYPE_OP_MAP_PARAMS_SETS,
        },
        ("test_to_dtype", "test_to_dtype_cpu"): {
            "param_sets": TO_DTYPE_OP_PARAMS_SETS,
            "expect_fail": TO_DTYPE_OP_EXPECT_FAIL,
        },
        ("test_round_trip_to_dtype", "test_round_trip_to_dtype_cpu"): {
            "ops_dict": {"add": torch.add},
            "param_sets": TO_DTYPE_OP_ROUND_TRIP_PARAMS_SETS,
            "expect_fail": TO_DTYPE_OP_ROUND_TRIP_EXPECT_FAIL,
        },
        ("test_conv2d", "test_conv2d_cpu"): {
            "param_sets": {
                "1x3x32_ksize3_no_pad": (
                    cached_randn((1, 3, 32, 32)),
                    cached_randn((16, 3, 3, 3)),
                    None,
                    (0, 0),
                    (1, 1),
                    1,
                ),
                "1x3x64_ksize3_pad1": (
                    cached_randn((1, 3, 64, 64)),
                    cached_randn((16, 3, 3, 3)),
                    None,
                    (1, 1),
                    (1, 1),
                    1,
                ),
                "2x3x32_ksize1": (
                    cached_randn((2, 3, 32, 32)),
                    cached_randn((8, 3, 1, 1)),
                    None,
                    (0, 0),
                    (1, 1),
                    1,
                ),
                "1x16x64_ksize3_pad1": (
                    cached_randn((1, 16, 64, 64)),
                    cached_randn((32, 16, 3, 3)),
                    None,
                    (1, 1),
                    (1, 1),
                    1,
                ),
                "1x64_ksize3_depthwise": (
                    cached_randn((1, 64, 32, 32)),
                    cached_randn((64, 1, 3, 3)),
                    None,
                    (1, 1),
                    (1, 1),
                    64,
                ),
                "mistral_model": (
                    cached_randn((1, 3, 392, 532)),
                    cached_randn((1024, 3, 14, 14)),
                    None,
                    (0, 0),
                    (1, 1),
                    1,
                ),
                "2x32_ksize1_stride2": (
                    cached_randn((2, 32, 64, 64)),
                    cached_randn((16, 32, 1, 1)),
                    None,
                    (0, 0),
                    (2, 2),
                    1,
                ),
                "1x3x128_ksize5": (
                    cached_randn((1, 3, 128, 128)),
                    cached_randn((8, 3, 5, 5)),
                    None,
                    (2, 2),
                    (1, 1),
                    1,
                ),
                "8x64_ksize3_pad1": (
                    cached_randn((8, 64, 128, 128)),
                    cached_randn((64, 1, 3, 3)),
                    None,
                    (1, 1),
                    (1, 1),
                    64,
                ),
            },
        },
        ("test_repeat", "test_repeat_cpu"): {
            "param_sets": {
                "1d_1": (cached_randn((64), dtype=torch.float16), 1),
                "1d_1_int32": (torch.randint(0, 100, (64,), dtype=torch.int32), 1),
                "1d_1_size1": (cached_randn((1), dtype=torch.float16), 1),
                "1d_1_size1_int32": (torch.randint(0, 100, (1,), dtype=torch.int32), 1),
                "2d_3x2": (cached_randn((2, 64), dtype=torch.float16), 3, 2),
                "2d_4x6": (cached_randn((2, 64), dtype=torch.float16), 4, 6),
                "2d_1x1": (cached_randn((2, 64), dtype=torch.float16), 1, 1),
                "3d_8x6x4": (cached_randn((2, 3, 64), dtype=torch.float16), 8, 6, 4),
            },
        },
        ("test_unfold", "test_unfold_cpu"): {
            "param_sets": {
                # 1D: Basic cases
                "1d_step1": (0, 3, 1, torch.arange(10, dtype=torch.float16)),
                "1d_step2": (0, 3, 2, torch.arange(20, dtype=torch.float16)),
                "1d_no_overlap": (0, 4, 4, torch.arange(16, dtype=torch.float16)),
                "1d_large": (0, 10, 5, cached_randn((100,))),
                # 2D: Different dimensions
                "2d_dim0": (0, 3, 1, cached_randn((10, 8))),
                "2d_dim1": (1, 4, 2, cached_randn((8, 16))),
                "2d_dim_neg": (-1, 3, 1, cached_randn((8, 12))),
                "2d_square": (0, 4, 2, cached_randn((8, 8))),
                # 3D: Each dimension
                "3d_dim0": (0, 3, 1, cached_randn((10, 8, 6))),
                "3d_dim1": (1, 4, 2, cached_randn((8, 16, 6))),
                "3d_dim2": (2, 3, 1, cached_randn((8, 6, 10))),
                # 4D: Deep learning typical
                "4d_batch": (0, 3, 1, cached_randn((8, 4, 6, 6))),
                "4d_spatial": (2, 3, 1, cached_randn((4, 8, 12, 6))),
                "4d_cnn": (2, 3, 1, cached_randn((2, 64, 28, 28))),
                # Edge cases
                "edge_window_1": (0, 1, 1, cached_randn((10,))),
                "edge_single_window": (0, 5, 1, torch.arange(5, dtype=torch.float16)),
                "edge_large_step": (0, 3, 7, cached_randn((30,))),
                "edge_pow2_64": (0, 16, 8, cached_randn((64,))),
                "edge_nopad_37": (0, 7, 3, cached_randn((37,))),
                "edge_nopad_2d": (1, 5, 2, cached_randn((16, 33))),
            },
        },
        ("test_unbind", "test_unbind_cpu"): {
            "param_sets": {
                # 1D — produces 0-D scalar tensors
                "1d_dim0": (0, cached_randn((8,))),
                # 2D — unbind along each axis
                "2d_dim0": (0, cached_randn((4, 64))),
                "2d_dim1": (1, cached_randn((4, 64))),
                "2d_dimneg1": (-1, cached_randn((4, 64))),
                # 3D — all three axes, including negative index
                "3d_dim0": (0, cached_randn((4, 8, 64))),
                "3d_dim1": (1, cached_randn((4, 8, 64))),
                "3d_dim2": (2, cached_randn((4, 8, 64))),
                "3d_dimneg1": (-1, cached_randn((4, 8, 64))),
                # 4D — innermost and non-innermost axes
                "4d_dim0": (0, cached_randn((2, 4, 8, 64))),
                "4d_dim3": (3, cached_randn((2, 4, 8, 64))),
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compare_with_cpu(self, *args, **kwargs):
        return utils_inductor.compare_with_cpu(*args, **kwargs)

    def compare(self, *args, **kwargs):
        return utils_inductor.compare(*args, **kwargs)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings("ignore:Backend Spyre does not support int64")
    def test_scalar_comparison_base(self, scalar, x):
        """Base method for testing equality comparison with scalar constants"""

        def fn(tensor, scalar_val):
            return tensor == scalar_val

        _compare_op_with_cpu(fn, None, x, scalar)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings("ignore:Backend Spyre does not support int64")
    def test_scalar_multidim_base(self, scalar, x):
        """Base method for testing equality comparison on multi-dimensional tensors"""

        def fn(tensor, scalar_val):
            return tensor == scalar_val

        _compare_op_with_cpu(fn, None, x, scalar)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings("ignore:Backend Spyre does not support int64")
    def test_scalar_vs_tensor_base(self, x, y, scalar):
        """Base method for testing both scalar and tensor comparisons work"""

        def fn(tensor_x, tensor_y, scalar_val):
            scalar_result = tensor_x == scalar_val
            tensor_result = tensor_x == tensor_y
            return scalar_result, tensor_result

        _compare_op_with_cpu(fn, None, x, y, scalar)

    def test_scalar_comparison(self):
        self.test_eq_scalar_int_42()

    def test_eq_scalar_constant_int(self):
        self.test_eq_scalar_int_10()

    def test_eq_scalar_constant_float(self):
        self.test_eq_scalar_float_3_5()

    def test_eq_scalar_constant_negative(self):
        self.test_eq_scalar_negative_5()

    def test_eq_scalar_constant_zero(self):
        self.test_eq_scalar_zero()

    def test_eq_scalar_constant_multidim(self):
        self.test_eq_scalar_multidim_2d()

    def test_eq_scalar_constant_large_tensor(self):
        self.test_eq_scalar_multidim_large()

    def test_eq_scalar_vs_tensor_comparison(self):
        self.test_eq_scalar_vs_tensor_mixed()

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_unary_op(self, op, x):
        if op == torch.reciprocal:
            # TODO: Division by 0 or near-zero differs on Spyre from CPU, sidestep for now.
            tiny_value_mask = torch.abs(x) < FP16_EPS
            x[tiny_value_mask] = FP16_EPS
        elif op == torch.floor:
            # To avoid cpu mismatch due to a negative fp16 having a fraction 0b0000000001
            x = x.to("spyre").cpu()

        self.compare_with_cpu(op, x)

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

        self.compare_with_cpu(fn, *tensor_args)

    def test_unary_op_cpu(self, op, x):
        self.compare_with_cpu(op, x)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_fallback_unary_op_cpu(self, op, x):
        self.compare_with_cpu(op, x)

    def test_binary_op(self, op, a, b):
        if op == torch.div:
            # TODO: Division by 0 or near-zero differs on Spyre from CPU, sidestep for now.
            tiny_value_mask = torch.abs(b) < FP16_EPS
            b[tiny_value_mask] = FP16_EPS

        self.compare_with_cpu(op, a, b)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_fallback_binary_op_cpu(self, op, x, y):
        self.compare_with_cpu(op, x, y, run_eager=False)

    # Increased mm test tolerance for splitk
    def test_mm_relaxed(self, op, a, b):
        K = b.shape[-2]
        if K > (128 // b.element_size()):  # multiple sticks
            self.compare_with_cpu(op, a, b, atol=0.1, rtol=0.1)
        else:  # single stick, no need to relax
            self.compare_with_cpu(op, a, b)

    def test_mm_autocast_cpu(self, enabled, a, b):
        def fn(a, b):
            with torch.autocast(device_type="spyre", enabled=enabled):
                return a @ b

        self.compare_with_cpu(fn, a, b)

    def test_binary_op_cpu(self, op, x, y):
        # Eager mode support varies by op:
        # - torch.eq, torch.ge, torch.gt, torch.lt: work eagerly
        # - torch.matmul: numerical divergence (close=False) in eager 2d case
        eager_supported = op in (
            torch.eq,
            torch.ge,
            torch.gt,
            torch.lt,
            torch.ne,
            torch.le,
        )
        self.compare_with_cpu(op, x, y, run_eager=eager_supported)

    def test_cmp_scalar_int64_cpu(self, op, x, scalar):
        # Test comparison ops with int64 tensors and scalar values.
        self.compare_with_cpu(op, x, scalar, run_eager=True, run_compile=False)

    def test_linear_fn(self, x, weight, bias):
        # NOTE: relaxing atol from 2e-1 to 3e-1 for multi-dim work division, single element fails without
        self.compare_with_cpu(
            torch.nn.functional.linear, x, weight, bias, atol=3e-1, rtol=2e-1
        )

    # Example where base function is not parameterized
    def test_add_broadcast_cpu(self, x, y):
        self.compare_with_cpu(lambda x, y: torch.add(x[None, :], y), x, y)

    def test_addmm_cpu(self, input, mat1, mat2):
        # NOTE: relaxing atol from 2e-1 to 3e-1 for multi-dim work division
        self.compare_with_cpu(torch.addmm, input, mat1, mat2, atol=3e-1, rtol=2e-1)

    def test_matmul_tiled_y(self):
        # Inspired by granite code that broke with no covering tests.
        # GQA pattern: y is a 5D contiguous buffer from clone(expand(...))
        # giving tiled host coords where the reduction dim decomposes as
        # floor(...) and Mod(...) over a single loop variable.
        B, H_KV, GQA, S, D = 2, 8, 4, 128, 128
        H = H_KV * GQA

        def fn(x, kv_cache):
            y = kv_cache.view(B, S, H_KV, D)
            y = y.permute(0, 2, 1, 3)
            y = y.unsqueeze(2)
            y = y.expand(-1, -1, GQA, -1, -1)
            y = y.clone()
            return torch.bmm(
                x.reshape(B * H, S, D),
                y.reshape(B * H, S, D).transpose(1, 2),
            )

        x = torch.randn(B, H, S, D, dtype=torch.float16)
        kv = torch.randn(B, S * H_KV, D, dtype=torch.float16)
        self.compare_with_cpu(fn, x, kv, atol=0.5, rtol=0.1)

    def test_matmul_tiled_x(self):
        # Inspired by granite code that broke with no covering tests.
        # x is a 4D contiguous buffer [B,S,H,D] giving tiled host coords
        # where the reduction dim decomposes as floor(...) and Mod(...)
        # over a single flat loop variable.
        B, S, H, D = 2, 128, 32, 128

        def fn(x_base, y):
            x = x_base.clone()
            return torch.matmul(x.reshape(B, S, H * D), y)

        x = torch.randn(B, S, H, D, dtype=torch.float16) * 0.01
        y = torch.randn(H * D, H * D, dtype=torch.float16) * 0.01
        self.compare_with_cpu(fn, x, y, atol=0.5, rtol=0.1)

    def test_matmul_1d_view_x(self):
        # x is a 1D buffer viewed as 2D: inductor keeps the 1D buffer and uses
        # a compound index, so reduction var must be found via symbol-set arithmetic.
        A, B, C = 64, 128, 256

        def fn(x, y):
            return x.view(A, B) @ y

        x = torch.rand(A * B, dtype=torch.float16) * 0.01
        y = torch.rand(B, C, dtype=torch.float16) * 0.01
        self.compare_with_cpu(fn, x, y, atol=0.5, rtol=0.1)

    def test_matmul_1d_view_y(self):
        # y is a 1D buffer viewed as 2D: same compound-index case but on y.
        A, B, C = 64, 128, 256

        def fn(x, y):
            return x @ y.view(B, C)

        x = torch.rand(A, B, dtype=torch.float16) * 0.01
        y = torch.rand(B * C, dtype=torch.float16) * 0.01
        self.compare_with_cpu(fn, x, y, atol=0.5, rtol=0.1)

    def test_matmul_1d_view_xy(self):
        # Both x and y are 1D buffers viewed as 2D.
        A, B, C = 64, 128, 256

        def fn(x, y):
            return x.view(A, B) @ y.view(B, C)

        x = torch.rand(A * B, dtype=torch.float16) * 0.01
        y = torch.rand(B * C, dtype=torch.float16) * 0.01
        self.compare_with_cpu(fn, x, y, atol=0.5, rtol=0.1)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings("ignore:Backend Spyre does not support int64")
    def test_reduce_cpu(self, op, x):
        self.compare_with_cpu(lambda x: op(x), x)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings("ignore:Backend Spyre does not support int64")
    def test_reduce_keepdim0_cpu(self, op, dim: int, x):
        # torch.max returns a tuple (values, indices); keep just the values tensor.
        if op == torch.max:
            self.compare_with_cpu(
                lambda x: op(x, dim=dim, keepdim=False)[0],
                x,
                run_eager=False,
                cpu_compile=True,
            )
        elif op == torch.min:
            _compare_op_with_cpu(lambda x: op(x, dim=dim, keepdim=False)[0], op, x)
        else:
            _compare_op_with_cpu(lambda x: op(x, dim=dim, keepdim=False), op, x)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings("ignore:Backend Spyre does not support int64")
    def test_reduce_keepdim1_cpu(self, op, dim: int, x):
        # torch.max returns a tuple (values, indices); keep just the values tensor.
        if op == torch.max:
            self.compare_with_cpu(
                lambda x: op(x, dim=dim, keepdim=True)[0],
                x,
                run_eager=False,
                cpu_compile=True,
            )
        elif op == torch.min:
            _compare_op_with_cpu(lambda x: op(x, dim=dim, keepdim=True)[0], op, x)
        else:
            _compare_op_with_cpu(lambda x: op(x, dim=dim, keepdim=True), op, x)

    def test_reduce_multidim_keepdim0_cpu(self, op, dims: tuple[int, ...], x):
        _compare_op_with_cpu(lambda x: op(x, dim=dims, keepdim=False), op, x)

    def test_reduce_multidim_keepdim1_cpu(self, op, dims: tuple[int, ...], x):
        _compare_op_with_cpu(lambda x: op(x, dim=dims, keepdim=True), op, x)

    def test_tuple_reduce_keepdim0_cpu(self, op, dim, x):
        _compare_op_with_cpu(lambda x: op(x, dim=dim, keepdim=False), op, x)

    def test_tuple_reduce_keepdim1_cpu(self, op, dim, x):
        _compare_op_with_cpu(lambda x: op(x, dim=dim, keepdim=True), op, x)

    def test_norm_keepdim0_cpu(self, op, ord, dim, x):
        _compare_op_with_cpu(lambda x: op(x, ord=ord, dim=dim, keepdim=False), op, x)

    def test_norm_keepdim1_cpu(self, op, ord, dim, x):
        _compare_op_with_cpu(lambda x: op(x, ord=ord, dim=dim, keepdim=True), op, x)

    def _get_core_reduction_invalid_dim_cases(self):
        x = cached_randn((3, 5, 64))
        ops = CORE_REDUCTION_OPS_DICT
        shared_cases = {
            "single_dim_oob_positive": lambda op, x: op(x, dim=4, keepdim=False),
            "single_dim_oob_negative": lambda op, x: op(x, dim=-4, keepdim=False),
            "duplicate_dims_tuple": lambda op, x: op(x, dim=(1, 1), keepdim=False),
            # After normalization, 2 and -1 alias the same dimension on a 3D tensor.
            "duplicate_dims_after_normalization_tuple": lambda op, x: op(
                x, dim=(2, -1), keepdim=False
            ),
            "multidim_oob_positive_tuple": lambda op, x: op(
                x, dim=(1, 4), keepdim=False
            ),
            "multidim_oob_negative_tuple": lambda op, x: op(
                x, dim=(1, -4), keepdim=False
            ),
        }
        api_only_cases = {
            "single_dim_non_integer_float": lambda op, x: op(x, dim=1.5, keepdim=False),
            "single_dim_non_integer_string": lambda op, x: op(
                x, dim="1", keepdim=False
            ),
            "multidim_non_integer_float": lambda op, x: op(
                x, dim=(1, 1.5), keepdim=False
            ),
            "multidim_non_integer_string": lambda op, x: op(
                x, dim=(1, "2"), keepdim=False
            ),
            "multidim_non_integer_none": lambda op, x: op(
                x, dim=(1, None), keepdim=False
            ),
            "multidim_invalid_container_set": lambda op, x: op(
                x, dim={1, 2}, keepdim=False
            ),
        }
        return x, ops, shared_cases, api_only_cases

    def _get_single_dim_reduction_invalid_dim_cases(self):
        x = cached_randn((3, 5, 64), dtype=torch.float32)
        ops = {
            "min": torch.min,
            "aminmax": torch.aminmax,
        }
        shared_cases = {
            "single_dim_oob_positive": lambda op, x: op(x, dim=3, keepdim=False),
            "single_dim_oob_negative": lambda op, x: op(x, dim=-4, keepdim=False),
        }
        api_only_cases = {
            "tuple_dim_not_supported": lambda op, x: op(x, dim=(1, 2), keepdim=False),
            "single_dim_non_integer_float": lambda op, x: op(x, dim=1.5, keepdim=False),
            "single_dim_non_integer_string": lambda op, x: op(
                x, dim="1", keepdim=False
            ),
        }
        return x, ops, shared_cases, api_only_cases

    def test_core_reduction_invalid_dims_api(self):
        x, ops, shared_cases, api_only_cases = (
            self._get_core_reduction_invalid_dim_cases()
        )

        for op_name, op in ops.items():
            for case_name, case_fn in {**shared_cases, **api_only_cases}.items():
                with self.subTest(op=op_name, case=case_name):
                    with pytest.raises(Exception) as exc_info:
                        case_fn(op, x)
                    print(
                        f"{op_name}/{case_name}: "
                        f"{exc_info.type.__name__}: {exc_info.value!r}"
                    )

    def test_core_reduction_invalid_dims_spyre(self):
        x, ops, shared_cases, _ = self._get_core_reduction_invalid_dim_cases()

        for op_name, op in ops.items():
            for case_name, case_fn in shared_cases.items():
                with self.subTest(op=op_name, case=case_name):
                    with pytest.raises(Exception) as exc_info:
                        _compile_and_run(
                            lambda x, _op=op, _case_fn=case_fn: _case_fn(_op, x),
                            (x,),
                            "spyre",
                        )
                    print(
                        f"{op_name}/{case_name}: "
                        f"{exc_info.type.__name__}: {exc_info.value!r}"
                    )

    def test_single_dim_reduction_invalid_dims_api(self):
        x, ops, shared_cases, api_only_cases = (
            self._get_single_dim_reduction_invalid_dim_cases()
        )

        for op_name, op in ops.items():
            for case_name, case_fn in {**shared_cases, **api_only_cases}.items():
                with self.subTest(op=op_name, case=case_name):
                    with pytest.raises(Exception) as exc_info:
                        case_fn(op, x)
                    print(
                        f"{op_name}/{case_name}: "
                        f"{exc_info.type.__name__}: {exc_info.value!r}"
                    )

    def test_single_dim_reduction_invalid_dims_spyre(self):
        x, ops, shared_cases, _ = self._get_single_dim_reduction_invalid_dim_cases()

        for op_name, op in ops.items():
            for case_name, case_fn in shared_cases.items():
                with self.subTest(op=op_name, case=case_name):
                    with pytest.raises(Exception) as exc_info:
                        _compile_and_run(
                            lambda x, _op=op, _case_fn=case_fn: _case_fn(_op, x),
                            (x,),
                            "spyre",
                        )
                    print(
                        f"{op_name}/{case_name}: "
                        f"{exc_info.type.__name__}: {exc_info.value!r}"
                    )

    def test_topk_cpu(self, x, k: int, dim: int):
        # torch.topk returns (values, indices); only compare values since
        # index tie-breaking can differ between backends.
        # aten::topk is not registered for Spyre eager dispatch.
        self.compare_with_cpu(
            lambda x: torch.topk(x, k, dim=dim)[0], x, run_eager=False
        )

    def test_min_tuple_output_keepdim0(self):
        x = unique_randn_along_dim((5, 7), dim=1)
        self.compare_with_cpu(
            lambda x: torch.min(x, dim=1, keepdim=False),
            x,
            run_eager=False,
        )

    def test_argmin_keepdim0(self):
        x = unique_randn_along_dim((5, 7), dim=1)
        self.compare_with_cpu(
            lambda x: torch.argmin(x, dim=1, keepdim=False),
            x,
            run_eager=False,
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.count_nonzero on "
            "floating inputs yet (stable error signature: Unsupported: "
            "unexpected argument Constant(value=0.0, dtype=torch.float16) to "
            "notequal)"
        ),
        strict=True,
    )
    def test_count_nonzero_float_dim0_known_xfail(self):
        x = cached_randn((67, 256))
        self.compare_with_cpu(
            lambda x: torch.count_nonzero(x, dim=0),
            x,
            run_eager=False,
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.count_nonzero on "
            "bool inputs yet (stable error signature: Unsupported: unexpected "
            "argument PointwiseOp(op='to_dtype', ...) to reduction lowering)"
        ),
        strict=True,
    )
    def test_count_nonzero_bool_dim0_known_xfail(self):
        x = torch.tensor(
            [
                [True, False, True, False, True, False, True],
                [False, False, True, True, False, True, False],
                [True, True, False, False, True, False, False],
                [False, True, False, True, False, True, True],
                [True, False, False, True, True, False, True],
            ],
            dtype=torch.bool,
        )
        self.compare_with_cpu(
            lambda x: torch.count_nonzero(x, dim=0),
            x,
            run_eager=False,
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend hits an internal lowering bug for "
            "torch.logsumexp (stable error signature: InductorError: "
            "IndexError: list index out of range)"
        ),
        strict=True,
    )
    def test_logsumexp_keepdim0_known_xfail(self):
        x = cached_randn((67, 256), scale=0.1)
        self.compare_with_cpu(
            lambda x: torch.logsumexp(x, dim=0, keepdim=False),
            x,
            run_eager=False,
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend hits an internal lowering bug for "
            "torch.nanmean (stable error signature: InductorError: IndexError: "
            "list index out of range)"
        ),
        strict=True,
    )
    def test_nanmean_all_dims_known_xfail(self):
        x = torch.tensor(
            [
                [float("nan"), 1.0, -2.0, 3.0],
                [4.0, float("nan"), -5.0, 6.0],
                [7.0, 8.0, float("nan"), -9.0],
            ],
            dtype=torch.float32,
        )
        self.compare_with_cpu(lambda x: torch.nanmean(x), x, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend hits an internal lowering bug for "
            "torch.nansum (stable error signature: InductorError: IndexError: "
            "list index out of range)"
        ),
        strict=True,
    )
    def test_nansum_all_dims_known_xfail(self):
        x = torch.tensor(
            [
                [float("nan"), 1.0, -2.0, 3.0],
                [4.0, float("nan"), -5.0, 6.0],
                [7.0, 8.0, float("nan"), -9.0],
            ],
            dtype=torch.float32,
        )
        self.compare_with_cpu(lambda x: torch.nansum(x), x, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.all yet (stable "
            "error signature: InductorError: AttributeError: "
            "'UnimplementedOp' object has no attribute 'iteration_space')"
        ),
        strict=True,
    )
    def test_all_dim0_known_xfail(self):
        x = torch.tensor(
            [
                [True, False, True, False],
                [True, True, False, False],
                [False, True, True, False],
            ],
            dtype=torch.bool,
        )
        self.compare_with_cpu(
            lambda x: torch.all(x, dim=0, keepdim=False), x, run_eager=False
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.any yet (stable "
            "error signature: InductorError: AttributeError: "
            "'UnimplementedOp' object has no attribute 'iteration_space')"
        ),
        strict=True,
    )
    def test_any_dim0_known_xfail(self):
        x = torch.tensor(
            [
                [True, False, True, False],
                [True, True, False, False],
                [False, True, True, False],
            ],
            dtype=torch.bool,
        )
        self.compare_with_cpu(
            lambda x: torch.any(x, dim=0, keepdim=False), x, run_eager=False
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.prod yet (stable "
            "error signature: InductorError: AttributeError: "
            "'UnimplementedOp' object has no attribute 'iteration_space')"
        ),
        strict=True,
    )
    def test_prod_dim0_known_xfail(self):
        x = cached_randn((67, 256), scale=0.1)
        self.compare_with_cpu(
            lambda x: torch.prod(x, dim=0, keepdim=False), x, run_eager=False
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.std yet (stable "
            "error signature: InductorError: TypeError: "
            "'UnimplementedOp' object is not subscriptable)"
        ),
        strict=True,
    )
    def test_std_dim0_known_xfail(self):
        x = cached_randn((67, 256), dtype=torch.float32)
        self.compare_with_cpu(
            lambda x: torch.std(x, dim=0, keepdim=False), x, run_eager=False
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.var yet (stable "
            "error signature: InductorError: TypeError: "
            "'UnimplementedOp' object is not subscriptable)"
        ),
        strict=True,
    )
    def test_var_dim0_known_xfail(self):
        x = cached_randn((67, 256), dtype=torch.float32)
        self.compare_with_cpu(
            lambda x: torch.var(x, dim=0, keepdim=False), x, run_eager=False
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.std_mean yet "
            "(stable error signature: InductorError: TypeError: "
            "'UnimplementedOp' object is not subscriptable)"
        ),
        strict=True,
    )
    def test_std_mean_dim0_known_xfail(self):
        x = cached_randn((67, 256), dtype=torch.float32)
        self.compare_with_cpu(
            lambda x: torch.std_mean(x, dim=0, keepdim=False), x, run_eager=False
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.var_mean yet "
            "(stable error signature: InductorError: TypeError: "
            "'UnimplementedOp' object is not subscriptable)"
        ),
        strict=True,
    )
    def test_var_mean_dim0_known_xfail(self):
        x = cached_randn((67, 256), dtype=torch.float32)
        self.compare_with_cpu(
            lambda x: torch.var_mean(x, dim=0, keepdim=False), x, run_eager=False
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.cumprod yet "
            "(stable error signature: NotImplementedError: Could not run "
            "'aten::cumprod.out' with arguments from the 'spyre' backend)"
        ),
        strict=True,
    )
    def test_cumprod_dim0_known_xfail(self):
        x = cached_randn((67, 256), scale=0.1)
        self.compare_with_cpu(lambda x: torch.cumprod(x, dim=0), x, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.logcumsumexp yet "
            "(stable error signature: NotImplementedError: Could not run "
            "'aten::_logcumsumexp' with arguments from the 'spyre' backend)"
        ),
        strict=True,
    )
    def test_logcumsumexp_dim0_known_xfail(self):
        x = cached_randn((67, 256), scale=0.1)
        self.compare_with_cpu(
            lambda x: torch.logcumsumexp(x, dim=0), x, run_eager=False
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.cummax yet "
            "(stable error signature: NotImplementedError: Could not run "
            "'aten::_cummax_helper' with arguments from the 'spyre' backend)"
        ),
        strict=True,
    )
    def test_cummax_dim0_known_xfail(self):
        x = unique_randn_along_dim((67, 256), dim=0)
        self.compare_with_cpu(lambda x: torch.cummax(x, dim=0), x, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.cummin yet "
            "(stable error signature: NotImplementedError: Could not run "
            "'aten::_cummin_helper' with arguments from the 'spyre' backend)"
        ),
        strict=True,
    )
    def test_cummin_dim0_known_xfail(self):
        x = unique_randn_along_dim((67, 256), dim=0)
        self.compare_with_cpu(lambda x: torch.cummin(x, dim=0), x, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.quantile yet "
            "(stable error signature: InductorError: Unsupported: unexpected "
            "argument PointwiseOp(op='to_dtype', ...) to mul)"
        ),
        strict=True,
    )
    def test_quantile_q050_dim0_known_xfail(self):
        x = cached_randn((67, 256), dtype=torch.float32)
        self.compare_with_cpu(
            lambda x: torch.quantile(x, 0.5, dim=0, keepdim=False),
            x,
            run_eager=False,
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.nanquantile yet "
            "(stable error signature: InductorError: IndexError: list index "
            "out of range)"
        ),
        strict=True,
    )
    def test_nanquantile_q050_dim0_known_xfail(self):
        x = torch.tensor(
            [
                [float("nan"), 1.0, -2.0, 3.0],
                [4.0, float("nan"), -5.0, 6.0],
                [7.0, 8.0, float("nan"), -9.0],
                [2.0, 3.0, 4.0, 5.0],
            ],
            dtype=torch.float32,
        )
        self.compare_with_cpu(
            lambda x: torch.nanquantile(x, 0.5, dim=0, keepdim=False),
            x,
            run_eager=False,
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.median yet "
            "(stable error signature: NotImplementedError: Could not run "
            "'aten::median.dim_values' with arguments from the 'spyre' backend)"
        ),
        strict=True,
    )
    def test_median_dim1_known_xfail(self):
        x = unique_randn_along_dim((67, 71, 256), dim=1)
        self.compare_with_cpu(
            lambda x: torch.median(x, dim=1, keepdim=False), x, run_eager=False
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.nanmedian yet "
            "(stable error signature: NotImplementedError: Could not run "
            "'aten::median.dim_values' with arguments from the 'spyre' backend)"
        ),
        strict=True,
    )
    def test_nanmedian_dim0_known_xfail(self):
        x = torch.tensor(
            [
                [float("nan"), 1.0, -2.0, 3.0],
                [4.0, float("nan"), -5.0, 6.0],
                [7.0, 8.0, float("nan"), -9.0],
            ],
            dtype=torch.float32,
        )
        self.compare_with_cpu(
            lambda x: torch.nanmedian(x, dim=0, keepdim=False),
            x,
            run_eager=False,
        )

    @pytest.mark.xfail(
        reason=(
            "Spyre compiled backend does not support torch.mode yet "
            "(stable error signature: NotImplementedError: Could not run "
            "'aten::mode' with arguments from the 'spyre' backend)"
        ),
        strict=True,
    )
    def test_mode_dim1_known_xfail(self):
        x = torch.tensor(
            [
                [0.0, 0.0, 2.0, 3.0],
                [1.0, 1.0, 4.0, 5.0],
                [2.0, 2.0, 6.0, 7.0],
            ],
            dtype=torch.float16,
        )
        self.compare_with_cpu(
            lambda x: torch.mode(x, dim=1, keepdim=False), x, run_eager=False
        )

    def test_max_sub_broadcast(self, dim: int, x):
        def fn(x):
            x_max = torch.max(x, dim=dim).values
            z = x - torch.unsqueeze(x_max, dim=dim)
            return z

        self.compare_with_cpu(fn, x)

    def test_t_1d_cpu(self, x):
        self.compare_with_cpu(lambda x: x.t(), x)

    def test_t_1d_contiguous_cpu(self, x):
        # Note: .contiguous() causes issues with eager mode, see https://github.com/torch-spyre/torch-spyre/issues/1149
        self.compare_with_cpu(lambda x: x.t().contiguous(), x, run_eager=False)

    def test_t_2d_cpu(self, x):
        self.compare_with_cpu(lambda x: x.t(), x)

    def test_t_2d_contiguous_cpu(self, x):
        # Note: .contiguous() causes issues with eager mode, see https://github.com/torch-spyre/torch-spyre/issues/1149
        self.compare_with_cpu(lambda x: x.t().contiguous(), x, run_eager=False)

    def test_transpose_2d_cpu(self, dim0: int, dim1: int, x):
        self.compare_with_cpu(lambda x: torch.transpose(x, dim0, dim1), x)

    def test_transpose_2d_contiguous_cpu(self, dim0: int, dim1: int, x):
        self.compare_with_cpu(lambda x: torch.transpose(x, dim0, dim1).contiguous(), x)

    def test_transpose_3d_cpu(self, dim0: int, dim1: int, x):
        self.compare_with_cpu(lambda x: torch.transpose(x, dim0, dim1), x)

    def test_transpose_3d_contiguous_cpu(self, dim0: int, dim1: int, x):
        self.compare_with_cpu(lambda x: torch.transpose(x, dim0, dim1).contiguous(), x)

    def test_transpose_4d_cpu(self, dim0: int, dim1: int, x):
        self.compare_with_cpu(lambda x: torch.transpose(x, dim0, dim1), x)

    def test_transpose_4d_contiguous_cpu(self, dim0: int, dim1: int, x):
        self.compare_with_cpu(lambda x: torch.transpose(x, dim0, dim1).contiguous(), x)

    def test_where_cpu(self, cond_op, x, y):
        # aten::where.self is not registered for the Spyre backend
        self.compare_with_cpu(
            lambda x, y: torch.where(cond_op(x, y), x, y), x, y, run_eager=False
        )

    def test_range_op(self, op, input, min, max, err):
        self.compare_with_cpu(lambda x: op(x, min, max), input, atol=err, rtol=err)

    def test_activation_cls(self, op, input, kwargs, err):
        # Spyre activation custom ops (e.g. spyre::gelu) have a pass-through
        # implementation that returns None in eager mode; they only work inside
        # torch.compile where the inductor lowering handles them
        self.compare_with_cpu(
            lambda x: op(**kwargs)(x), input, atol=err, rtol=err, run_eager=False
        )

    def test_activation_fn(self, op, input, err):
        self.compare_with_cpu(lambda x: op(x), input, atol=err, rtol=err)

    @pytest.mark.filterwarnings(
        "ignore:Backend Spyre does not support int64:UserWarning"
    )
    def test_clone(self, x):
        # Eager clone + .cpu() causes heap corruption (invalid fastbin / corrupted
        # double-linked list) in libsenlib for fp16/fp32 small tensors, and SIGBUS
        # for bool tensors.  Disable eager mode for all dtypes.
        self.compare_with_cpu(lambda a: torch.clone(a).contiguous(), x, run_eager=False)

    def test_permute(self, input_dims, dims):
        self.compare_with_cpu(
            lambda input: torch.permute(input, dims),
            cached_randn(input_dims, dtype=torch.float16),
        )

    @pytest.mark.filterwarnings(
        "ignore:torch\\.ops\\.spyre\\.overwrite is deprecated.*:FutureWarning"
    )
    def test_overwrite_cpu(self, input, output, dims, offsets):
        def fn(input, output):
            torch.ops.spyre.overwrite(input, output, dims, offsets)
            return output

        self.compare_with_cpu(fn, input, output, clone_inputs=True)

    def test_flatten_cpu(self, start_dim, end_dim, x):
        """Test flatten operation with various dimension ranges."""
        self.compare_with_cpu(lambda x: x.flatten(start_dim, end_dim), x)

    def test_dropout_functional(self, input, kwargs):
        self.compare_with_cpu(lambda a: torch.nn.functional.dropout(a, **kwargs), input)

    def test_inplace_op_cpu(self, op, dst, src):
        def fn(dst, src):
            dst = dst.clone()
            result = op(dst, src)
            assert id(result) == id(dst)
            return result

        # Eager mode hangs/crashes when executing inplace operations on Spyre tensors
        self.compare_with_cpu(fn, dst, src, run_eager=False)

    def test_inplace_copy_noncontiguous_cpu(self, dst, src):
        def fn(dst, src):
            dst_t = dst.t()
            dst_t.copy_(src)
            return dst_t.contiguous()

        self.compare_with_cpu(fn, dst, src, run_eager=False)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_fallback_cpu(self, x):
        def fn(t):
            t = torch.exp(t)  # compiled op
            t = torch.sin(t)  # fallback op
            t = torch.exp(t)  # compiled op
            return t

        with pytest.warns(UserWarning) as record:
            self.compare_with_cpu(fn, x, cpu_compile=True)

        print(f"Warn {len(record)}")

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_arange_cpu(self, *args):
        def fn(device=None):
            return torch.arange(*args, dtype=torch.float16, device=device)

        self.compare_with_cpu(fn, needs_device=True)

    def test_empty_like_cpu(self, x):
        def fn(x):
            y = torch.empty_like(x)
            y.fill_(1.0)
            return y

        self.compare_with_cpu(fn, x)

    def test_empty_like_dtype_override_cpu(self, x):
        """Test empty_like with dtype override (fp16->fp32 or fp32->fp16)."""
        # Determine target dtype (opposite of input)
        target_dtype = torch.float32 if x.dtype == torch.float16 else torch.float16

        def fn(x):
            y = torch.empty_like(x, dtype=target_dtype)
            y.fill_(1.0)
            return y

        self.compare_with_cpu(fn, x)

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
        self.compare_with_cpu(fn, x, run_eager=False)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_new_ones_cpu(self, x, y):
        self.compare_with_cpu(lambda x: x.new_ones((x.size())), x)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_ones_cpu(self, size):
        """Compiled torch.ones(size) on Spyre (identity broadcast) matches CPU."""

        def fn(device=None):
            return torch.ones(size, dtype=torch.float16, device=device)

        self.compare_with_cpu(fn, needs_device=True, cpu_compile=False)

    def test_numel_cpu(self, x):
        self.compare_with_cpu(lambda x: torch.numel(x), x)

    def test_cat_cpu(self, dim, *tensors):
        def fn(*tensors):
            return torch.cat(tensors, dim=dim)

        self.compare_with_cpu(fn, *tensors)

    def test_pad_cpu(self, x, pad):
        """Compiled torch.nn.functional.pad (constant zero) on Spyre matches CPU."""

        def fn(x):
            return torch.nn.functional.pad(x, pad)

        self.compare_with_cpu(fn, x)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_full_cpu(self, *args):
        def fn(device=None):
            return torch.full(*args, dtype=torch.float16, device=device)

        self.compare_with_cpu(fn, needs_device=True, cpu_compile=False)

    def test_dim_op_cpu(self, op, dim, *args):
        def fn(*args):
            return op(dim, *args)

        # Combined ops (exp+squeeze, exp+unsqueeze, add+unsqueeze) fail in eager
        # because the eager exp/add dispatch internally triggers torch.compile on
        # shapes that the Spyre backend compiler cannot handle
        self.compare_with_cpu(fn, *args, run_eager=False)

    def test_dim_op_cpu_eager(self, op, dim, *args):
        def fn(*args):
            return op(dim, *args)

        # Simple dim ops (softmax, squeeze, unsqueeze, sum+squeeze) work in eager
        self.compare_with_cpu(fn, *args)

    def test_attention_cpu(self, *args):
        def fn(q, k, v, sm_scale):
            qk = q @ k.transpose(-1, -2).contiguous()
            p = qk.softmax(dim=-1) * sm_scale
            return p @ v

        # mm/bmm on Spyre tensors segfaults in libsenlib without the torch.compile
        # execution context that normally initialises the hardware session
        self.compare_with_cpu(fn, *args, run_eager=False)

    def test_layernorm_cpu(self, input, weight, bias):
        def fn(input, weight, bias):
            return torch.nn.functional.layer_norm(
                input, input.shape[1:], weight=weight, bias=bias
            )

        self.compare_with_cpu(fn, input, weight, bias)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_rmsnorm_cpu(self, x):
        def fn(input):
            return torch.nn.functional.rms_norm(input, [input.shape[-1]], eps=1e-6)

        self.compare_with_cpu(fn, x)

    def test_softplus_cpu(self, x):
        beta = 1.0
        threshold = 20.0

        def fn(input):
            return torch.nn.functional.softplus(input, beta, threshold)

        self.compare_with_cpu(fn, x)

    # --- Migrated from test_ops.py ---

    def test_copy_roundtrip(self, x):
        self.compare_with_cpu(lambda x: x, x)

    def test_mean_default_cpu(self, x):
        self.compare_with_cpu(lambda x: torch.mean(x), x)

    def test_mean_cpu(self, dim, keepdim, x):
        self.compare_with_cpu(lambda x: torch.mean(x, dim=dim, keepdim=keepdim), x)

    def test_zeros_cpu(self, size):
        def fn(device=None):
            return torch.zeros(*size, dtype=torch.float16, device=device)

        self.compare_with_cpu(fn, needs_device=True, cpu_compile=False)

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

        self.compare_with_cpu(
            fn,
            x,
            run_compile=(execution_mode == "compiled"),
            run_eager=(execution_mode == "eager"),
        )

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_addmm_scaled_cpu(self, alpha, input, mat1, mat2):
        self.compare_with_cpu(
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

        self.compare_with_cpu(fn, input, mat1, mat2, atol=2e-1, rtol=2e-1)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_embedding_cpu(self, indices, weight, padding_idx):
        self.compare_with_cpu(
            lambda indices, weight: torch.nn.functional.embedding(
                indices, weight, padding_idx=padding_idx
            ),
            indices,
            weight,
        )

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_isin_cpu(self, elements, test_elements):
        self.compare_with_cpu(torch.isin, elements, test_elements)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_isin_out_cpu(self, elements, test_elements):
        def fn(elements, test_elements):
            out = torch.empty(elements.shape, dtype=torch.bool, device=elements.device)
            torch.isin(elements, test_elements, out=out)
            return out

        self.compare_with_cpu(fn, elements, test_elements)

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

    def test_random_from_cpu(self):
        """Test that tensor.random_(-5, 5) fills a tensor with random values in [-5, 5)."""
        gen = torch.manual_seed(42)
        x_spyre = torch.zeros(3, 5, dtype=torch.float16, device="spyre")
        y_cpu = torch.zeros(3, 5, dtype=torch.float16, device="cpu")
        y_cpu.random_(-5, 5, generator=gen)
        gen.manual_seed(42)
        x_spyre.random_(-5, 5, generator=gen)
        x_cpu = x_spyre.to("cpu")

        assert torch.all(x_cpu >= -5) and torch.all(x_cpu < 5), (
            f"random_ values out of range [-5, 5): {x_cpu}"
        )
        assert not torch.all(x_cpu == x_cpu[0]), "random_ produced all identical values"
        torch.testing.assert_close(x_cpu, y_cpu, rtol=0.0, atol=0.0)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_tril_cpu(self, x):
        def fn(input):
            return torch.tril(input)

        self.compare_with_cpu(fn, x)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_triu_cpu(self, x, diagonal):
        def fn(input, diagonal):
            return torch.triu(input, diagonal)

        self.compare_with_cpu(fn, x, diagonal)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_sdpa_cpu(self, q, k, v, attn_mask, is_causal, enable_gqa):
        def fn(q, k, v, attn_mask, is_causal, enable_gqa):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask, is_causal=is_causal, enable_gqa=enable_gqa
            )

        self.compare_with_cpu(fn, q, k, v, attn_mask, is_causal, enable_gqa)

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

            self.compare_with_cpu(fn, x, cpu_compile=False)

        elif len(args) == 2:
            x, y = args

            def fn(a, b):
                result = a * b
                return result.item()

            self.compare_with_cpu(fn, x, y, cpu_compile=False)

    def test_split_cpu(self, op, dim, index, x):
        def fn(x):
            return op(dim, index, x)

        self.compare_with_cpu(fn, x, clone_inputs=True, run_eager=False)

    def test_slice_cpu(self, op, dim, index, x):
        def fn(x):
            start = index * (x.size()[dim] // 3)
            end = (index + 1) * (x.size()[dim] // 3)
            if dim == 0:
                return op(dim, x[start:end])
            elif dim == 1:
                return op(dim, x[:, start:end])
            elif dim == 2:
                return op(dim, x[:, :, start:end])

        self.compare_with_cpu(fn, x, clone_inputs=True, run_eager=False)

    def test_slice_synthetic_dims_cpu(self, x):
        def fn(x):
            return x[:, 1:2, :, :, :] + x[:, :, 2:3, :, :]

        self.compare_with_cpu(fn, x, clone_inputs=True, run_eager=False)

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

        self.compare_with_cpu(fn, q, freqs, cpu_compile=False)

    def test_sum_eager(self, op, dim: int, keepdim: bool, x):
        self.compare_with_cpu(lambda x: op(x, dim=dim, keepdim=keepdim), x)

    def test_mean_eager(self, op, dim: int, keepdim: bool, x):
        self.compare_with_cpu(lambda x: op(x, dim=dim, keepdim=keepdim), x)

    def test_max_eager(self, op, dim: int, keepdim: bool, x):
        self.compare_with_cpu(lambda x: op(x, dim=dim, keepdim=keepdim)[0], x)

    def test_min_eager(self, op, dim: int, keepdim: bool, x):
        self.compare_with_cpu(lambda x: op(x, dim=dim, keepdim=keepdim)[0], x)

    def test_where_eager_default_fallback(self, op, condition):
        self.compare_with_cpu(lambda condition: op(condition), condition)

    def test_where_eager(self, op, condition, x, y):
        self.compare_with_cpu(
            lambda condition, x, y: op(condition, x, y), condition, x, y
        )

    def test_where_eager_scalar(self, op, condition, x, y):
        x = torch.tensor(x, dtype=torch.float16)
        y = torch.tensor(y, dtype=torch.float16)
        self.compare_with_cpu(
            lambda condition, x, y: op(condition, x, y), condition, x, y
        )

    def test_where_eager_selfout(self, op, condition, x, y, z):
        self.compare_with_cpu(
            lambda condition, x, y, z: op(condition, x, y, out=z), condition, x, y, z
        )

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
        self.compare_with_cpu(fn, q, k, v, cpu_compile=False, run_eager=False)

    def test_to_dtype_op_map(self, src, dst):
        result = DtypeOpTable.get_operator(src, dst)
        conversions = DtypeOpTable.get_table()
        if (src, dst) in conversions:
            expected = conversions[(src, dst)]
            assert result == expected, (
                f"Expected {expected} for {src}->{dst}, got {result}"
            )
        else:
            assert result is None, (
                f"Expected None for unsupported {src}->{dst}, got {result}"
            )

    def test_to_dtype_cpu(self, x, dst_dtype):
        def fn(x, dst_dtype):
            return x.to(dtype=dst_dtype)

        self.compare_with_cpu(
            fn,
            x,
            dst_dtype,
            cpu_compile=False,
            run_eager=False,
        )

    def test_round_trip_to_dtype_cpu(self, op, x, dst_dtype):
        def fn(op, x, dst_dtype):
            y = x.to(dst_dtype)
            z = op(y, y)
            return z.to(x.dtype)

        self.compare_with_cpu(
            fn,
            op,
            x,
            dst_dtype,
            cpu_compile=False,
            run_eager=False,
        )

    def test_bool_conversion_from_spyre(self):
        torch.manual_seed(42)

        def test_fn(input):
            tmp = torch.mul(input, input)
            tmp = tmp.to(dtype=torch.bool)
            return tmp

        input_cpu = (torch.randn(64) > 0.0).to(dtype=torch.float16)
        expected_output = test_fn(input_cpu)

        input_spyre = input_cpu.to("spyre")
        compiled_fn = torch.compile(test_fn, backend="inductor")
        output_spyre = compiled_fn(input_spyre)
        output_cpu = output_spyre.cpu()

        assert torch.equal(output_cpu, expected_output), (
            f"Bool conversion failed: got {output_cpu.sum().item()}/{64} True, expected {expected_output.sum().item()}/{64}"
        )

    def test_conv2d_cpu(self, x, weight, bias, padding, stride, groups):
        def fn(x, weight, bias, padding, stride, groups):
            return torch.conv2d(
                x, weight, bias, stride=stride, padding=padding, groups=groups
            )

        self.compare_with_cpu(
            fn,
            x,
            weight,
            bias,
            padding,
            stride,
            groups,
            atol=0.5,
            rtol=0.1,
        )

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_index_copy_cpu(self):
        """Test torch.index_copy operation on Spyre matches CPU in eager mode.

        Note: index_copy creates layout incompatibilities in compiled mode due to
        its scatter pattern, so we only test eager mode execution.
        """

        def fn(dst, dim, index, src):
            # Use non-mutating version
            return torch.index_copy(dst, dim, index, src)

        # Test case 1: Basic 2D tensor, copy along dim 0
        dst1 = torch.randn(5, 3)
        index1 = torch.tensor([0, 2, 4])
        src1 = torch.randn(3, 3)
        # Only run in eager mode - compiled mode has layout issues with scatter ops
        self.compare_with_cpu(
            fn, dst1, 0, index1, src1, run_compile=False, run_eager=True
        )

        # Test case 2: Copy along dim 1
        dst2 = torch.randn(3, 5)
        index2 = torch.tensor([1, 3])
        src2 = torch.randn(3, 2)
        self.compare_with_cpu(
            fn, dst2, 1, index2, src2, run_compile=False, run_eager=True
        )

        # Test case 3: 3D tensor
        dst3 = torch.randn(4, 3, 2)
        index3 = torch.tensor([0, 2])
        src3 = torch.randn(2, 3, 2)
        self.compare_with_cpu(
            fn, dst3, 0, index3, src3, run_compile=False, run_eager=True
        )

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_index_copy_inplace_prefill(self):
        """Test Tensor.index_copy_ prefill pattern from Ministral-3-14B-Instruct-2512.

        Prefill: writes 14 tokens into KV-cache at positions 0-13.
          cache:  [1, 8, 2048, 128] float16
          dim:    2
          index:  [14] int64 (arange 0-13)
          source: [1, 8, 14, 128] float16
        """

        def index_copy_fn(cache, index, source):
            cache = cache.clone()
            result = cache.index_copy_(2, index, source)
            assert result.data_ptr() == cache.data_ptr(), (
                "index_copy_: return value is not the same tensor as self"
            )
            return result

        cache = cached_randn((1, 8, 2048, 128))
        index = torch.arange(14, dtype=torch.int64)
        source = cached_randn((1, 8, 14, 128), differentiation="prefill")

        self.compare_with_cpu(
            index_copy_fn, cache, index, source, run_compile=False, run_eager=True
        )

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_index_copy_inplace_decode(self):
        """Test Tensor.index_copy_ decode pattern from Ministral-3-14B-Instruct-2512.

        Decode: writes 1 token into KV-cache at position 14.
          cache:  [1, 8, 2048, 128] float16
          dim:    2
          index:  [1] int64 (contains 14)
          source: [1, 8, 1, 128] float16
        """

        def index_copy_fn(cache, index, source):
            cache = cache.clone()
            result = cache.index_copy_(2, index, source)
            assert result.data_ptr() == cache.data_ptr(), (
                "index_copy_: return value is not the same tensor as self"
            )
            return result

        cache = cached_randn((1, 8, 2048, 128))
        index = torch.tensor([14], dtype=torch.int64)
        source = cached_randn((1, 8, 1, 128), differentiation="decode")

        self.compare_with_cpu(
            index_copy_fn, cache, index, source, run_compile=False, run_eager=True
        )

    def test_repeat_cpu(self, x, *repeat_args):
        def fn(a):
            return a.repeat(*repeat_args)

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_unfold_cpu(self, dimension, size, step, x):
        """Test unfold operation (view only, no contiguous).

        NOTE: .contiguous() is not tested as it currently fails.
        This test verifies the unfold VIEW operation works correctly.
        """
        self.compare_with_cpu(lambda x: x.unfold(dimension, size, step), x)

    def test_unbind_cpu(self, dim: int, x):
        self.compare_with_cpu(lambda a: torch.unbind(a, dim=dim), x)

    @pytest.mark.xfail(
        reason=(
            "RESTICKIFY_OP does not support FP32 dtype "
            "(stable error signature: Unsupported: ReStickifyOpHBM on DataFormats.IEEE_FP32)"
        ),
        strict=True,
    )
    def test_restickify_fp32_unsupported_xfail(self):
        """Verify RESTICKIFY_OP correctly rejects FP32 dtype.

        Operations that would trigger restickify (like transpose + pointwise)
        should fail with Unsupported error when using FP32 tensors.
        """
        x = torch.randn((128, 128), dtype=torch.float32)
        y = torch.randn((128, 128), dtype=torch.float32)
        # Transpose creates layout incompatibility that triggers restickify
        self.compare_with_cpu(
            lambda x, y: x.t() + y,
            x,
            y,
            run_eager=False,
        )

    @pytest.mark.xfail(
        reason=(
            "RESTICKIFY_OP does not support INT64 dtype "
            "(stable error signature: Unsupported: ReStickifyOpHBM on DataFormats.INT32)"
        ),
        strict=True,
    )
    def test_restickify_int64_unsupported_xfail(self):
        """Verify RESTICKIFY_OP correctly rejects INT64 dtype.

        Operations that would trigger restickify (like transpose + pointwise)
        should fail with Unsupported error when using INT64 tensors.
        """
        x = torch.randint(0, 100, (128, 128), dtype=torch.int64)
        y = torch.randint(0, 100, (128, 128), dtype=torch.int64)
        # Transpose creates layout incompatibility that triggers restickify
        self.compare_with_cpu(
            lambda x, y: x.t() + y,
            x,
            y,
            run_eager=False,
        )


if __name__ == "__main__":
    unittest.main()
