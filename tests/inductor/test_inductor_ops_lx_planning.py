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
    "test_bitwise_and_bitwise_and_bool_1d",
    "test_bitwise_and_bitwise_and_bool_2d",
    "test_bitwise_and_bitwise_and_bool_3d",
    "test_bitwise_and_bitwise_and_bool_4d",
    "test_bitwise_and_bitwise_and_int_1d",
    "test_bitwise_and_bitwise_and_int_2d",
    "test_bitwise_and_bitwise_and_int_3d",
    "test_bitwise_and_bitwise_and_int_4d",
    "test_bitwise_not_bitwise_not_bool_1d",
    "test_bitwise_not_bitwise_not_bool_2d",
    "test_bitwise_not_bitwise_not_bool_3d",
    "test_bitwise_not_bitwise_not_bool_4d",
    "test_bitwise_not_bitwise_not_int_1d",
    "test_bitwise_not_bitwise_not_int_2d",
    "test_bitwise_not_bitwise_not_int_3d",
    "test_bitwise_not_bitwise_not_int_4d",
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
    "test_cat_4d_dim3_fp32",
    "test_cat_4d_dim3",
    "test_clone_bool_1d",
    "test_clone_bool_2d",
    "test_clone_bool_3d",
    "test_cmp_eq_1d",
    "test_cmp_eq_2d",
    "test_cmp_eq_3d",
    "test_cmp_eq_broadcast",
    "test_cmp_ge_1d",
    "test_cmp_ge_2d",
    "test_cmp_ge_3d",
    "test_cmp_ge_broadcast",
    "test_cmp_gt_1d",
    "test_cmp_gt_2d",
    "test_cmp_gt_3d",
    "test_cmp_gt_broadcast",
    "test_cmp_le_1d",
    "test_cmp_le_2d",
    "test_cmp_le_3d",
    "test_cmp_le_broadcast",
    "test_cmp_lt_1d",
    "test_cmp_lt_2d",
    "test_cmp_lt_3d",
    "test_cmp_lt_broadcast",
    "test_cmp_ne_1d",
    "test_cmp_ne_2d",
    "test_cmp_ne_3d",
    "test_cmp_ne_broadcast",
    "test_einsum_einsum_67x255_255x128",
    "test_einsum_einsum_67x256_256x128",
    "test_fallback_1d",
    "test_fallback_2d",
    "test_fallback_3d",
    "test_full_value_1",
    "test_full_value_2",
    "test_inplace_copy_copy_bool",
    "test_isin_out_tensor_tensor",
    "test_isin_tensor_tensor",
    "test_large_matmul_matmul_3d_M3_K11_N2880",
    "test_large_matmul_matmul_4d_B2_H2_M2048_K2048_N65536",
    "test_linear_2d_bias",
    "test_linear_2d_no_bias",
    "test_linear_3d_bias",
    "test_linear_3d_no_bias",
    "test_logical_not_logical_not_1d_bool",
    "test_logical_not_logical_not_1d_fp16",
    "test_logical_not_logical_not_2d_bool",
    "test_logical_not_logical_not_2d_fp16",
    "test_logical_not_logical_not_3d_bool",
    "test_logical_not_logical_not_3d_fp16",
    "test_logical_not_logical_not_4d_bool",
    "test_logical_not_logical_not_4d_fp16",
    "test_logical_not_logical_not_bool_single_elem",
    "test_logical_not_logical_not_fp16_single_elem",
    "test_max_keepdim0_max_2d_dim_0_int64",
    "test_max_keepdim0_max_2d_dim_0",
    "test_max_keepdim0_max_2d_dim_1_int64",
    "test_max_keepdim0_max_2d_dim_1",
    "test_max_keepdim0_max_3d_dim_0",
    "test_max_keepdim0_max_3d_dim_1",
    "test_max_keepdim0_max_3d_dim_2",
    "test_max_keepdim0_max_4d_dim_0",
    "test_max_keepdim0_max_4d_dim_1",
    "test_max_keepdim0_max_4d_dim_2",
    "test_max_keepdim0_max_4d_dim_3",
    "test_max_keepdim0_max_4d_dim_gpt0",
    "test_max_keepdim0_max_4d_dim_gpt1",
    "test_max_keepdim1_max_2d_dim_0_int64",
    "test_max_keepdim1_max_2d_dim_1_int64",
    "test_mm_mm_55x2_2x99",
    "test_mm_mm_67x67_67x67",
    "test_pointwise_binary_op_sub_67x71x256_67x71x256",
    "test_pointwise_unary_op_exp_67x71x256",
    "test_qkv_attn_paths_fms_decode_gqa",
    "test_rope_fms_prefill_bs1",
    "test_rope_fms_prefill",
    "test_sdpa_gqa_prefill_causal",
    "test_sdpa_gqa_prefill",
    "test_sdpa_mha_prefill_causal",
    "test_sdpa_mha_prefill",
    "test_sdpa_mha_prefill_mask",
    "test_softmax_softmax_3d_dim0",
    "test_softmax_softmax_3d_dim1",
    "test_softmax_softmax_3d_dim2",
    "test_squeeze_reduction_sum_3d0",
    "test_squeeze_reduction_sum_4d0",
    "test_squeeze_single_3d0",
    "test_squeeze_single_4d0",
    "test_t_2d_49159x4096",
    "test_t_2d_contiguous_4096x49280",
    "test_t_2d_contiguous_49280x4096",
    "test_topk_2d_k4_dim_minusone",
    "test_transpose_2d_large_dim_0_1",
    "test_transpose_2d_large_dim_0_1_nopad",
    "test_transpose_2d_large_dim_0_2",
    "test_transpose_2d_large_dim_0_2_nopad",
    "test_transpose_2d_large_dim_1_2",
    "test_transpose_2d_large_dim_1_2_nopad",
]


class LxPlanningTwoOpPointwiseAdditionTest(unittest.TestCase):
    def wrap_pointwise(self, fn):
        @functools.wraps(fn)
        def make_seq_of_ops(*fn_args, **fn_kwargs):
            result = fn(*fn_args, **fn_kwargs)
            return pytree.tree_map(
                lambda x: x + x if isinstance(x, torch.Tensor) else x, result
            )

        return make_seq_of_ops

    def compare_with_cpu(self, fn, *args, **kwargs):
        def source_check(source):
            FileCheck().check("{lx: 0}").run(source)

        kwargs["cpu_compile"] = False
        return compare_with_cpu(
            self.wrap_pointwise(fn), source_check=source_check, *args, **kwargs
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
            self.wrap_pointwise(fn),
            *args,
            atol=cpu_atol,
            rtol=cpu_rtol,
            needs_device=needs_device,
            cpu_compile=False,
        )


copy_tests(
    make_lx_planning_class(inductor.test_inductor_ops.TestOps),
    LxPlanningTwoOpPointwiseAdditionTest,
    "lx_planning_pointwise",
    POINTWISE_TEST_FAILURES if not tests_lx_planning_run_skips else None,
)


REDUCTION_TEST_FAILURES = [
    "test_addmm_out_basic",
    "test_addmm_scaled_alpha_0_5",
    "test_copy_roundtrip_2d",
    "test_alias_operands_cube_67x71x256",
    "test_alias_operands_double_67x71x256",
    "test_alias_operands_triple_67x71x256",
    "test_amax_keepdim1_amax_scalar_tensor",
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
    "test_cat_4d_dim3_fp32",
    "test_cat_4d_dim3",
    "test_copy_roundtrip_4d_stick",
    "test_einsum_einsum_67x255_255x128",
    "test_einsum_einsum_67x256_256x128",
    "test_einsum_einsum_67x67_67x67",
    "test_fallback_1d",
    "test_fallback_2d",
    "test_fallback_3d",
    "test_full_value_1",
    "test_full_value_2",
    "test_large_matmul_matmul_2d_M2048_K2048_N65536",
    "test_large_matmul_matmul_3d2d_M3_K11_N2880",
    "test_large_matmul_matmul_4d_B2_H2_M2048_K2048_N65536",
    "test_layernorm_2d",
    "test_linear_2d_bias",
    "test_linear_2d_no_bias",
    "test_linear_3d_bias",
    "test_linear_3d_no_bias",
    "test_mm_mm_55x2_2x99",
    "test_permute_4d_0_3_1_2",
    "test_permute_4d_0_m2_m1_1",
    "test_pointwise_binary_op_add_67x71x256_67x71x256",
    "test_pointwise_binary_op_div_67x256_67x256",
    "test_pointwise_binary_op_div_67x71x256_67x71x256",
    "test_pointwise_binary_op_sub_67x71x256_67x71x256",
    "test_pointwise_range_op_clamp_fp16",
    "test_pointwise_unary_op_exp_67x71x256",
    "test_pointwise_unary_op_reciprocal_67x256",
    "test_pointwise_unary_op_reciprocal_67x71x256",
    "test_qkv_attn_paths_fms_decode_gqa",
    "test_rmsnorm_2d",
    "test_rope_fms_prefill_bs1",
    "test_rope_fms_prefill",
    "test_scalar_cpu_combined_3d",
    "test_scalar_cpu_combined_4d",
    "test_scalar_cpu_div_2d",
    "test_scalar_cpu_mul_2d",
    "test_scalar_cpu_true_divide_2d",
    "test_sdpa_gqa_prefill_causal",
    "test_sdpa_gqa_prefill",
    "test_sdpa_mha_prefill_causal",
    "test_sdpa_mha_prefill",
    "test_sdpa_mha_prefill_mask",
    "test_softmax_softmax_3d_dim0",
    "test_softmax_softmax_3d_dim1",
    "test_softmax_softmax_3d_dim2",
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
    "test_topk_2d_k4_dim_minusone",
    "test_transpose_2d_large_dim_0_1",
    "test_transpose_2d_large_dim_0_1_nopad",
    "test_transpose_2d_large_dim_0_2",
    "test_transpose_2d_large_dim_0_2_nopad",
    "test_transpose_2d_large_dim_1_2",
    "test_transpose_2d_large_dim_1_2_nopad",
]


class LxPlanningTwoOpReductionTest(unittest.TestCase):
    def wrap_reduction(self, fn):
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

    def compare_with_cpu(self, fn, *args, **kwargs):
        def source_check(source):
            FileCheck().check("{lx: 0}").run(source)

        kwargs["cpu_compile"] = False
        return compare_with_cpu(
            self.wrap_reduction(fn), source_check=source_check, *args, **kwargs
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
            self.wrap_reduction(fn),
            *args,
            atol=cpu_atol,
            rtol=cpu_rtol,
            needs_device=needs_device,
            cpu_compile=False,
        )


copy_tests(
    make_lx_planning_class(inductor.test_inductor_ops.TestOps),
    LxPlanningTwoOpReductionTest,
    "lx_planning_reduction",
    REDUCTION_TEST_FAILURES if not tests_lx_planning_run_skips else None,
)
