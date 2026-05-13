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

import functools
import torch_spyre
import os
import sys
import torch
from torch.utils import _pytree as pytree
from torch.testing import FileCheck

from torch._dynamo.testing import make_test_cls_with_patches

import unittest
from utils_inductor import compare_with_cpu, copy_tests

_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(_test_dir)

import inductor.test_inductor_ops  # noqa: E402

tests_lx_planning_run_skips: bool = (
    os.environ.get("TEST_LX_PLANNING_RUN_SKIPS", "0") == "1"
)


def make_lx_planning_class(cls):
    return make_test_cls_with_patches(
        cls,
        "LxPlanning",
        "",
        (torch_spyre._inductor.config, "lx_planning", True),
        (torch_spyre._inductor.config, "allow_all_ops_in_lx_planning", True),
        (torch_spyre._inductor.config, "sencores", 1),
    )


POINTWISE_TEST_FAILURES = [
    "test_addmm_1152_10x1152_1152x1152",
    "test_bmm_bmm_2x55x2_2x2x99",
    "test_bmm_bmm_2x99x65_2x65x55",
    "test_cat_1d_dim0",
    "test_cat_1d_dim0_three_tensors",
    "test_cat_2d_dim0_diff_size",
    "test_cat_2d_dim0_three_tensors",
    "test_cat_2d_dim1_diff_size",
    "test_cat_3d_dim0",
    "test_cat_3d_dim1",
    "test_cat_3d_dim1_size1",
    "test_cat_3d_dim2",
    "test_cat_4d_dim0",
    "test_cat_4d_dim1",
    "test_cat_4d_dim2",
    "test_cat_4d_dim3",
    "test_cat_4d_dim3_fp32",
    "test_einsum_einsum_55x2_2x99",
    "test_einsum_einsum_67x255_255x128",
    "test_einsum_einsum_67x256_256x128",
    "test_einsum_einsum_67x67_67x67",
    "test_fallback_1d",
    "test_fallback_2d",
    "test_fallback_3d",
    "test_full_value_1",
    "test_full_value_2",
    "test_large_matmul_matmul_3d_M3_K11_N2880",
    "test_large_matmul_matmul_4d_B2_H2_M2048_K2048_N65536",
    "test_matmul_matmul_2x3x55x2_2x3x2x99",
    "test_matmul_matmul_2x3x99x65_2x3x65x55",
    "test_matmul_matmul_2x55x2_2x2x99",
    "test_matmul_matmul_2x99x65_2x65x55",
    "test_matmul_matmul_55x2_2x99",
    "test_matmul_matmul_99x65_65x55",
    "test_mm_autocast_f16_disabled",
    "test_mm_autocast_f16_enabled",
    "test_mm_mm_55x2_2x99",
    "test_mm_mm_67x255_255x128",
    "test_mm_mm_67x67_67x67",
    "test_pad_2d_both_dims",
    "test_pad_2d_dim0_left",
    "test_pad_2d_dim0_left_only",
    "test_pad_2d_last_dim_left_and_right_stick_aligned",
    "test_pad_2d_last_dim_left_stick_aligned",
    "test_pad_2d_last_dim_left_two_sticks",
    "test_pad_2d_last_dim_right",
    "test_pad_3d_dim0_left",
    "test_pad_3d_dim1_left",
    "test_pad_3d_dim1_right",
    "test_pad_3d_last_dim_right",
    "test_pad_4d_dim0_left",
    "test_qkv_attn_paths_fms_decode_gqa",
    "test_squeeze_reduction_sum_3d0",
    "test_squeeze_reduction_sum_4d0",
    "test_squeeze_single_3d0",
    "test_squeeze_single_4d0",
    "test_t_2d_49159x4096",
    "test_t_2d_contiguous_4096x49280",
    "test_t_2d_contiguous_49280x4096",
    "test_topk_2d_k4_dim0",
    "test_transpose_2d_large_dim_0_1",
    "test_transpose_2d_large_dim_0_1_nopad",
    "test_transpose_2d_large_dim_0_2",
    "test_transpose_2d_large_dim_0_2_nopad",
    "test_transpose_2d_large_dim_1_2",
    "test_transpose_2d_large_dim_1_2_nopad",
    "test_transpose_4d_contiguous_dim_0_3",
    "test_transpose_4d_contiguous_dim_1_3",
]


class _LxPlanningTwoOpTestBase(unittest.TestCase):
    """Shared scaffolding for LX-planning two-op tests.

    Subclasses implement ``wrap(fn)`` to append a second op (pointwise,
    reduction, ...) onto the result of each upstream op test.
    """

    def wrap(self, fn):
        raise NotImplementedError

    def compare_with_cpu(self, fn, *args, **kwargs):
        def source_check(source):
            FileCheck().check("{lx: 0}").run(source)

        kwargs["cpu_compile"] = False
        return compare_with_cpu(
            self.wrap(fn), source_check=source_check, *args, **kwargs
        )

    def compare(
        self,
        fn,
        *args,
        atol=0.0,
        rtol=0.0,
        cpu_atol=0.1,
        cpu_rtol=0.1,
        needs_device=False,
    ):
        return compare_with_cpu(
            self.wrap(fn),
            *args,
            atol=cpu_atol,
            rtol=cpu_rtol,
            needs_device=needs_device,
            cpu_compile=False,
        )


class LxPlanningTwoOpPointwiseAdditionTest(_LxPlanningTwoOpTestBase):
    def wrap(self, fn):
        @functools.wraps(fn)
        def make_seq_of_ops(*fn_args, **fn_kwargs):
            result = fn(*fn_args, **fn_kwargs)
            return pytree.tree_map(
                lambda x: x + x
                if isinstance(x, torch.Tensor) and x.dtype == torch.float16
                else x,
                result,
            )

        return make_seq_of_ops


copy_tests(
    make_lx_planning_class(inductor.test_inductor_ops.TestOps),
    LxPlanningTwoOpPointwiseAdditionTest,
    "lx_planning_pointwise",
    POINTWISE_TEST_FAILURES if not tests_lx_planning_run_skips else None,
)


REDUCTION_TEST_FAILURES = [
    "test_addmm_out_basic",
    "test_addmm_scaled_alpha_0_5",
    "test_alias_operands_cube_67x71x256",
    "test_alias_operands_double_67x71x256",
    "test_alias_operands_triple_67x71x256",
    "test_bmm_bmm_2x55x2_2x2x99",
    "test_bmm_bmm_2x99x65_2x65x55",
    "test_cat_1d_dim0",
    "test_cat_1d_dim0_three_tensors",
    "test_cat_2d_dim0_diff_size",
    "test_cat_2d_dim0_three_tensors",
    "test_cat_2d_dim1_diff_size",
    "test_cat_3d_dim0",
    "test_cat_3d_dim1",
    "test_cat_3d_dim1_size1",
    "test_cat_3d_dim2",
    "test_cat_4d_dim0",
    "test_cat_4d_dim1",
    "test_cat_4d_dim2",
    "test_cat_4d_dim3",
    "test_cat_4d_dim3_fp32",
    "test_copy_roundtrip_3d",
    "test_copy_roundtrip_4d_stick",
    "test_einsum_einsum_55x2_2x99",
    "test_einsum_einsum_67x255_255x128",
    "test_einsum_einsum_67x256_256x128",
    "test_einsum_einsum_67x67_67x67",
    "test_fallback_1d",
    "test_fallback_2d",
    "test_fallback_3d",
    "test_full_value_1",
    "test_full_value_2",
    "test_large_matmul_matmul_2d_M2048_K2048_N65536",
    "test_large_matmul_matmul_3d_M3_K11_N2880",
    "test_large_matmul_matmul_4d_B2_H2_M2048_K2048_N65536",
    "test_linear_2d_bias",
    "test_linear_2d_no_bias",
    "test_matmul_matmul_2x3x55x2_2x3x2x99",
    "test_matmul_matmul_2x3x99x65_2x3x65x55",
    "test_matmul_matmul_2x55x2_2x2x99",
    "test_matmul_matmul_2x99x65_2x65x55",
    "test_matmul_matmul_55x2_2x99",
    "test_matmul_matmul_99x65_65x55",
    "test_mm_autocast_f16_disabled",
    "test_mm_autocast_f16_enabled",
    "test_mm_mm_55x2_2x99",
    "test_mm_mm_67x255_255x128",
    "test_mm_mm_67x67_67x67",
    "test_pad_2d_both_dims",
    "test_pad_2d_dim0_left",
    "test_pad_2d_dim0_left_only",
    "test_pad_2d_last_dim_left_and_right_stick_aligned",
    "test_pad_2d_last_dim_left_stick_aligned",
    "test_pad_2d_last_dim_left_two_sticks",
    "test_pad_2d_last_dim_right",
    "test_pad_3d_dim0_left",
    "test_pad_3d_dim1_left",
    "test_pad_3d_dim1_right",
    "test_pad_3d_last_dim_right",
    "test_pad_4d_dim0_left",
    "test_permute_4d_0_3_1_2",
    "test_permute_4d_0_m2_m1_1",
    "test_pointwise_binary_op_add_67x71x256_67x71x256",
    "test_pointwise_binary_op_div_67x256_67x256",
    "test_pointwise_binary_op_div_67x71x256_67x71x256",
    "test_pointwise_range_op_clamp_fp16",
    "test_pointwise_unary_op_reciprocal_67x256",
    "test_pointwise_unary_op_reciprocal_67x71x256",
    "test_qkv_attn_paths_fms_decode_gqa",
    "test_rmsnorm_2d",
    "test_scalar_cpu_combined_3d",
    "test_scalar_cpu_combined_4d",
    "test_scalar_cpu_div_2d",
    "test_scalar_cpu_mul_2d",
    "test_scalar_cpu_true_divide_2d",
    "test_split_split3_1d0s0",
    "test_split_split3_1d0s1",
    "test_split_split3_1d0s2",
    "test_split_split3_2d0s1",
    "test_split_split3_2d0s2",
    "test_split_split3_3d0s1",
    "test_split_split3_3d0s2",
    "test_squeeze_reduction_sum_3d0",
    "test_squeeze_reduction_sum_4d0",
    "test_squeeze_single_3d0",
    "test_squeeze_single_4d0",
    "test_t_2d_49159x4096",
    "test_t_2d_contiguous_1088x320",
    "test_t_2d_contiguous_320x320",
    "test_t_2d_contiguous_4096x49280",
    "test_t_2d_contiguous_49280x4096",
    "test_topk_2d_k4_dim0",
    "test_transpose_2d_large_dim_0_1",
    "test_transpose_2d_large_dim_0_1_nopad",
    "test_transpose_2d_large_dim_0_2",
    "test_transpose_2d_large_dim_0_2_nopad",
    "test_transpose_2d_large_dim_1_2",
    "test_transpose_2d_large_dim_1_2_nopad",
    "test_transpose_4d_contiguous_dim_0_3",
    "test_transpose_4d_contiguous_dim_1_3",
]


class LxPlanningTwoOpReductionTest(_LxPlanningTwoOpTestBase):
    def wrap(self, fn):
        @functools.wraps(fn)
        def make_seq_of_ops(*fn_args, **fn_kwargs):
            result = fn(*fn_args, **fn_kwargs)
            return pytree.tree_map(
                lambda x: torch.sum(x, dim=0)
                if isinstance(x, torch.Tensor) and x.dtype == torch.float16
                else x,
                result,
            )

        return make_seq_of_ops


copy_tests(
    make_lx_planning_class(inductor.test_inductor_ops.TestOps),
    LxPlanningTwoOpReductionTest,
    "lx_planning_reduction",
    REDUCTION_TEST_FAILURES if not tests_lx_planning_run_skips else None,
)

# Made with Bob
