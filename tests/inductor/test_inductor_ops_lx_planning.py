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

import copy
import functools
import torch_spyre
import os
import sys
import torch
from torch.utils import _pytree as pytree
from torch.testing import FileCheck

from torch._dynamo.testing import make_test_cls_with_patches

import unittest
from utils_inductor import compare_with_cpu

_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(_test_dir)

import inductor.test_inductor_ops  # noqa: E402

tests_lx_planning_run_skips: bool = (
    os.environ.get("TEST_LX_PLANNING_RUN_SKIPS", "0") == "1"
)

# By default, only run one representative test per (prefix, op) cell of
# TestOps.PARAMS plus all non-parameterized methods. Set this to "1" to
# wrap every generated test — useful for thorough triage, skip-list
# maintenance, and CI but slow for everyday dev workflow.
tests_lx_planning_full: bool = os.environ.get("TEST_LX_PLANNING_FULL", "0") == "1"


def make_lx_planning_class(cls):
    return make_test_cls_with_patches(
        cls,
        "LxPlanning",
        "",
        (torch_spyre._inductor.config, "lx_planning", True),
        (torch_spyre._inductor.config, "allow_all_ops_in_lx_planning", True),
        (torch_spyre._inductor.config, "sencores", 32),
    )


def _copy_inherited_methods(src, dst, attrs):
    for attr in attrs:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def _canonical_test_names(test_cls):
    """Pick one representative test name per (prefix, op) cell of
    ``test_cls.PARAMS``, plus all non-parameterized test methods.

    For a PARAMS entry with an ``ops_dict``, every op gets one shape (the
    first ``param_sets`` case); without an ``ops_dict``, only the first
    case is emitted.
    """
    params = getattr(test_cls, "PARAMS", {})
    canonical = set()
    parameterized_prefixes = []
    for (prefix, _base_func_name), cases in params.items():
        param_sets = cases.get("param_sets", {})
        if not param_sets:
            continue
        parameterized_prefixes.append(prefix)
        first_case = next(iter(param_sets))
        ops_dict = cases.get("ops_dict")
        if ops_dict:
            for op_name in ops_dict:
                canonical.add(f"{prefix}_{op_name}_{first_case}")
        else:
            canonical.add(f"{prefix}_{first_case}")
    for name in vars(test_cls):
        if name.startswith("test_") and not any(
            name.startswith(p + "_") for p in parameterized_prefixes
        ):
            canonical.add(name)
    return canonical


def _copy_canonical_tests(
    src_cls, dst_cls, suffix, test_failures, inherited_test_attributes
):
    """Copy test methods from ``src_cls`` into ``dst_cls`` with ``_{suffix}``
    appended to each name. Unless ``TEST_LX_PLANNING_FULL`` is set, restrict
    to the canonical subset derived from ``TestOps.PARAMS``."""
    keep = (
        None
        if tests_lx_planning_full
        else _canonical_test_names(inductor.test_inductor_ops.TestOps)
    )
    for name, value in src_cls.__dict__.items():
        if not name.startswith("test_"):
            continue
        if keep is not None and name not in keep:
            continue

        @functools.wraps(value)
        def new_test(self, value=value):
            return value(self)

        new_test.__dict__ = copy.deepcopy(value.__dict__)
        if test_failures and name in test_failures:
            new_test = unittest.skip("Skipped!")(new_test)
        setattr(dst_cls, f"{name}_{suffix}", new_test)
    _copy_inherited_methods(src_cls, dst_cls, inherited_test_attributes)


INHERITED_TEST_ATTRIBUTES = [
    "is_dtype_supported",
    "_get_core_reduction_invalid_dim_cases",
    "_get_single_dim_reduction_invalid_dim_cases",
]

POINTWISE_TEST_FAILURES = [
    "test_attention_3d",
    "test_attention_3d_batch_size_1",
    "test_attention_4d",
    "test_bitwise_not_bitwise_not_bool_4d",
    "test_cat_3d_dim0",
    "test_cat_3d_dim1",
    "test_cat_4d_dim0",
    "test_cat_4d_dim1",
    "test_cat_4d_dim2",
    "test_einsum_einsum_67x255_255x128",
    "test_einsum_einsum_67x256_256x128",
    "test_fallback_3d",
    # torch.flatten tests - Contiguous access pattern with span of 4 elements
    # within 64-wide padded stick not supported. Requires Mod(d0, ELEMS_PER_STICK)
    # support for partially-filled contiguous regions. See PR #1866.
    "test_flatten_2d_full",
    "test_flatten_3d_full",
    "test_flatten_3d_mixed_dims",
    "test_flatten_3d_neg_dims",
    "test_flatten_3d_neg_full",
    "test_flatten_3d_noncontig_full",
    "test_flatten_3d_noncontig_partial",
    "test_flatten_3d_trailing",
    "test_flatten_4d_full",
    "test_flatten_4d_large_full",
    "test_flatten_4d_trailing",
    "test_full_value_1",
    "test_full_value_2",
    "test_matmul_1d_view_x",
    "test_matmul_1d_view_xy",
    "test_matmul_matmul_55x2_2x99",
    "test_matmul_tiled_x",
    "test_matmul_tiled_y",
    "test_max_keepdim0_max_4d_dim_1",
    "test_max_keepdim0_max_4d_dim_2",
    "test_max_keepdim0_max_fp16_4d_dim_1",
    "test_max_keepdim0_max_fp16_4d_dim_2",
    "test_max_sub_broadcast_4d_dim_0",
    "test_max_sub_broadcast_4d_dim_1",
    "test_max_sub_broadcast_4d_dim_2",
    "test_max_sub_broadcast_4d_dim_3",
    "test_min_keepdim0_min_fp16_4d_dim_1",
    "test_min_keepdim0_min_fp16_4d_dim_2",
    "test_mm_autocast_f16_disabled",
    "test_mm_autocast_f16_enabled",
    "test_mm_mm_55x2_2x99",
    "test_pad_3d_dim0_left",
    "test_pad_3d_dim1_left",
    "test_pad_3d_dim1_right",
    "test_pad_3d_last_dim_negative_right",
    "test_pad_4d_dim0_left",
    "test_pad_4d_dim1_mixed",
    "test_pad_4d_dim2_negative_both",
    "test_pointwise_binary_op_add_7x12x32x64_7x12x32x64",
    "test_pointwise_binary_op_div_7x12x32x64_7x12x32x64",
    "test_pointwise_binary_op_maximum_7x12x32x64_7x12x32x64",
    "test_pointwise_binary_op_minimum_7x12x32x64_7x12x32x64",
    "test_pointwise_binary_op_mul_7x12x32x64_7x12x32x64",
    "test_pointwise_binary_op_sub_7x12x32x64_7x12x32x64",
    "test_qkv_attn_paths_fms_decode_gqa",
    "test_qkv_attn_paths_fms_fms_decode_gqa",
    "test_qkv_attn_paths_fms_prefill_gqa",
    "test_reduce_edge_multidim_keepdim0_sum_large_2d_dim_01_all",
    "test_reduce_edge_multidim_keepdim1_sum_large_2d_dim_01_all",
    "test_rmsnorm_3d",
    "test_rmsnorm_4d",
    "test_rope_fms_prefill",
    "test_rope_fms_prefill_bs1",
    "test_round_trip_to_dtype_add_float16_to_float32_2x4x8x64",
    "test_sdpa_gqa_prefill",
    "test_sdpa_gqa_prefill_causal",
    "test_sdpa_mha_prefill",
    "test_sdpa_mha_prefill_causal",
    "test_sdpa_mha_prefill_mask",
    "test_slice_add_3d2s0",
    "test_slice_add_3d2s1",
    "test_slice_add_3d2s2",
    "test_slice_copy__3d2s0",
    "test_slice_copy__3d2s1",
    "test_slice_copy__3d2s2",
    "test_slice_exp_3d2s0",
    "test_slice_exp_3d2s1",
    "test_slice_exp_3d2s2",
    "test_softmax_softmax_4d_dim0",
    "test_softmax_softmax_4d_dim1",
    "test_softmax_softmax_4d_dim2",
    "test_softmax_softmax_4d_dim3",
    "test_softplus_3d",
    "test_sub_broadcast_sub_3d_4d",
    "test_sum_keepdim0_sum_fp16_4d_dim_0",
    "test_sum_keepdim0_sum_fp16_4d_dim_1",
    "test_sum_keepdim0_sum_fp16_4d_dim_2",
    "test_to_dtype_bfloat16_to_float16_2x4x8x64",
    "test_transpose_2d_contiguous_dim_0_1",
    "test_transpose_2d_contiguous_dim_0_2",
    "test_transpose_2d_large_dim_0_1",
    "test_transpose_2d_large_dim_0_1_nopad",
    "test_transpose_2d_large_dim_0_2",
    "test_transpose_2d_large_dim_0_2_nopad",
    "test_transpose_3d_contiguous_dim_0_1",
    "test_transpose_3d_contiguous_dim_0_2",
    "test_transpose_4d_contiguous_dim_0_3",
    "test_transpose_4d_contiguous_dim_1_2",
    "test_transpose_4d_contiguous_dim_1_3",
    "test_tril_3d",
    "test_triu_3d",
    "test_unbind_1d_dim0",
    "test_vector_norm_keepdim0_vector_norm_ord2_3d_dim_12",
    "test_vector_norm_keepdim0_vector_norm_ord2_5d_dim_1234",
    "test_vector_norm_keepdim0_vector_norm_ord2_5d_mixed_1_neg1",
    "test_vector_norm_keepdim0_vector_norm_ordinf_4d_dim_neg1",
    "test_vector_norm_keepdim0_vector_norm_ordneginf_4d_dim_23",
    "test_vector_norm_keepdim1_vector_norm_ord2_3d_dim_12",
    "test_vector_norm_keepdim1_vector_norm_ord2_5d_dim_1234",
    "test_vector_norm_keepdim1_vector_norm_ord2_5d_mixed_1_neg1",
    "test_vector_norm_keepdim1_vector_norm_ordinf_4d_dim_neg1",
    "test_vector_norm_keepdim1_vector_norm_ordneginf_4d_dim_23",
    "test_where_self_out_where_fp16_2d",
]


class _LxPlanningTwoOpTestBase(unittest.TestCase):
    """Shared scaffolding for LX-planning two-op tests.

    Subclasses implement ``wrap(fn)`` to append a second op (pointwise,
    reduction, ...) onto the result of each upstream op test.
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

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


_copy_canonical_tests(
    make_lx_planning_class(inductor.test_inductor_ops.TestOps),
    LxPlanningTwoOpPointwiseAdditionTest,
    "lx_planning_pointwise",
    POINTWISE_TEST_FAILURES if not tests_lx_planning_run_skips else None,
    INHERITED_TEST_ATTRIBUTES,
)


REDUCTION_TEST_FAILURES = [
    "test_activation_cls_gelu_fp16",
    "test_addmm_1152_10x1152_1152x1152",
    "test_addmm_out_basic",
    "test_addmm_scaled_alpha_0_5",
    "test_alias_operands_cube_67x71x256",
    "test_alias_operands_double_67x71x256",
    "test_alias_operands_triple_67x71x256",
    "test_attention_3d",
    "test_bmm_bmm_2x55x2_2x2x99",
    "test_bmm_bmm_2x99x65_2x65x55",
    "test_copy_roundtrip_3d",
    "test_copy_roundtrip_4d_stick",
    "test_einsum_einsum_55x2_2x99",
    "test_einsum_einsum_67x255_255x128",
    "test_einsum_einsum_67x256_256x128",
    "test_einsum_einsum_67x67_67x67",
    "test_fallback_1d",
    "test_fallback_2d",
    "test_fallback_3d",
    # torch.flatten tests - Contiguous access pattern with span of 4 elements
    # within 64-wide padded stick not supported. Requires Mod(d0, ELEMS_PER_STICK)
    # support for partially-filled contiguous regions. See PR #1866.
    "test_flatten_3d_neg_dims",
    "test_flatten_3d_noncontig_partial",
    "test_flatten_3d_trailing",
    "test_flatten_4d_trailing",
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
    "test_pointwise_binary_op_int64_add_1d",
    "test_pointwise_binary_op_int64_maximum_1d",
    "test_pointwise_binary_op_int64_minimum_1d",
    "test_pointwise_binary_op_int64_mul_1d",
    "test_pointwise_binary_op_int64_sub_1d",
    "test_permute_4d_0_3_1_2",
    "test_permute_4d_0_m2_m1_1",
    "test_pointwise_binary_op_add_67x71x256_67x71x256",
    "test_pointwise_binary_op_div_67x256_67x256",
    "test_pointwise_binary_op_div_67x71x256_67x71x256",
    "test_pointwise_range_op_clamp_fp16",
    "test_pointwise_unary_op_reciprocal_67x256",
    "test_pointwise_unary_op_reciprocal_67x71x256",
    "test_qkv_attn_paths_fms_decode_gqa",
    "test_reduce_edge_multidim_keepdim0_sum_large_2d_dim_01_all",
    "test_reduce_edge_multidim_keepdim1_sum_large_2d_dim_01_all",
    "test_rmsnorm_2d",
    "test_rope_fms_prefill_bs1",
    "test_scalar_cpu_combined_3d",
    "test_scalar_cpu_combined_4d",
    "test_scalar_cpu_div_2d",
    "test_scalar_cpu_mul_2d",
    "test_scalar_cpu_true_divide_2d",
    "test_sdpa_mha_prefill",
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
    "test_unbind_1d_dim0",
    "test_transpose_2d_contiguous_dim_0_2",
    "test_transpose_2d_large_dim_0_1",
    "test_transpose_2d_large_dim_0_1_nopad",
    "test_transpose_2d_large_dim_0_2",
    "test_transpose_2d_large_dim_0_2_nopad",
    "test_transpose_2d_large_dim_1_2",
    "test_transpose_2d_large_dim_1_2_nopad",
    "test_transpose_3d_contiguous_dim_0_2",
    "test_transpose_4d_contiguous_dim_0_3",
    "test_transpose_4d_contiguous_dim_1_3",
    "test_where_self_out_where_fp16_2d",
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


_copy_canonical_tests(
    make_lx_planning_class(inductor.test_inductor_ops.TestOps),
    LxPlanningTwoOpReductionTest,
    "lx_planning_reduction",
    REDUCTION_TEST_FAILURES if not tests_lx_planning_run_skips else None,
    INHERITED_TEST_ATTRIBUTES,
)
