# Copyright 2026 The Torch-Spyre Authors.
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
from math import prod

import pytest
import torch
import torch._dynamo
import torch.nn.functional as F

import torch_spyre  # noqa: F401


MAX_CORES = 32
SEP = "=" * 100

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": getattr(torch, "float8_e4m3fn", torch.float16),
}


def _rand(shape, dtype_key):
    """Build a random tensor of the given dtype on the spyre device.

    torch.rand() has no fp8 kernel, so an fp8 request is built in fp16
    and cast down -- a raw type conversion, not Spyre's own
    quantize_fp8_with_scale (see issue #4310).
    """
    t = DTYPE_MAP[dtype_key]
    fp8_t = getattr(torch, "float8_e4m3fn", None)
    if fp8_t is not None and t is fp8_t:
        return torch.rand(*shape, dtype=torch.float16, device="spyre").to(t)
    return torch.rand(*shape, dtype=t, device="spyre")


class _WDTestCase(unittest.TestCase):
    """Shared base: resets dynamo before every test method, since each
    case compiles a fresh graph and stale cached state from an earlier
    case must not leak in."""

    def setUp(self):
        torch._dynamo.reset()


def cost_model_planner(splits):
    """Stand-in for the planner result; production returns this dynamically."""
    return dict(splits)


def apply_splits(planner_splits):
    """Stand-in for apply_splits(): stores the chosen plan on the operation."""
    return dict(planner_splits)


def assert_same(stage_a_name, stage_a, stage_b_name, stage_b, sizes):
    """Compare every dimension, not only the total number of cores."""
    if all(stage_a.get(dim, 1) == stage_b.get(dim, 1) for dim in sizes):
        return
    changed = {
        dim: (stage_a.get(dim, 1), stage_b.get(dim, 1))
        for dim in sizes
        if stage_a.get(dim, 1) != stage_b.get(dim, 1)
    }
    raise AssertionError(
        f"{stage_a_name} != {stage_b_name}: {stage_a} vs {stage_b} (changed: {changed})"
    )


def assert_valid(splits, sizes):
    """Generic plan checks; these do not hard-code a particular algorithm choice."""
    cores = prod(splits.values())
    assert cores <= MAX_CORES, f"uses {cores} cores, limit is {MAX_CORES}"
    for dim, size in sizes.items():
        split = splits.get(dim, 1)
        assert split >= 1, f"{dim} has invalid split {split}"
        assert size % split == 0, f"{dim} size {size} is not divisible by {split}"


class TestDotProduct1D(_WDTestCase):
    """1D vector/dot-product reference cases -- always Pass 3, no reduction
    dimension to route through the cost model."""

    def test_dot_1d_reference_baseline_fp16(self):
        """T000: reference baseline"""
        a = _rand((512,), "fp16")
        b = _rand((512,), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_dot_1d_llama_granite_hidden_dim_fp16(self):
        """T000b: Llama/Granite hidden-dim vector"""
        a = _rand((4096,), "fp16")
        b = _rand((4096,), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_dot_1d_llama_granite_hidden_dim_bf16(self):
        """T000c: bf16 hidden-dim vector"""
        a = _rand((4096,), "bf16")
        b = _rand((4096,), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_dot_1d_gptoss_hidden_dim_fp16(self):
        """T000d: gpt-oss-20b hidden-dim vector"""
        a = _rand((2880,), "fp16")
        b = _rand((2880,), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_dot_1d_mistral_hidden_dim_bf16(self):
        """T000e: Mistral hidden-dim vector"""
        a = _rand((5120,), "bf16")
        b = _rand((5120,), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestMatmul2D(_WDTestCase):
    """2D aten.mm -- fails Gate 1 (not BATCH_MATMUL_OP) before Work
    Division's own routing; lifted to a batch-of-1 3D bmm upstream."""

    def test_mm_2d_decode_linear_proj_fp16(self):
        """T010: decode linear proj"""
        a = _rand((1, 4096), "fp16")
        b = _rand((4096, 4096), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_2d_prefill_linear_proj_fp16(self):
        """T011: prefill linear proj"""
        a = _rand((2048, 4096), "fp16")
        b = _rand((4096, 4096), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_2d_decode_lmhead_vocab_bf16(self):
        """T012: decode lm_head vocab"""
        a = _rand((1, 4096), "bf16")
        b = _rand((4096, 32000), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_2d_prefill_lmhead_vocab_bf16(self):
        """T013: prefill lm_head vocab"""
        a = _rand((2048, 4096), "bf16")
        b = _rand((4096, 32000), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_2d_llama_mlp_upproj_fp16(self):
        """T014: Llama MLP up-proj"""
        a = _rand((2048, 4096), "fp16")
        b = _rand((4096, 11008), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_2d_decode_scorev_fp16(self):
        """T015: decode score x V (mm)"""
        a = _rand((1, 4096), "fp16")
        b = _rand((4096, 128), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #4310: raw-cast FP8 (no quantize_fp8_with_scale metadata) is unsupported by the FP8 matmul lowering"
    )
    def test_mm_2d_granite33_lmhead_fp8_expect_fail(self):
        """T01P: granite-3.3-8b lm_head"""
        a = _rand((1, 4096), "fp8")
        b = _rand((4096, 49159), "fp8")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestBmm3DGreedyPass3(_WDTestCase):
    """3D bmm, B=1 M=1 -> Gate 3 fires (row_dims empty) -> Pass 3 greedy.
    Includes the tracked tsp#4032 core-underutilization shape."""

    def test_bmm_3d_b1_m1_tiny_decode_fp16(self):
        """T040: tiny decode"""
        a = _rand((1, 1, 128), "fp16")
        b = _rand((1, 128, 64), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_narrow_n_fp16(self):
        """T041: narrow N"""
        a = _rand((1, 1, 128), "fp16")
        b = _rand((1, 128, 512), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_granite_decode_qkT_fp16(self):
        """T042: Granite decode QK^T"""
        a = _rand((1, 1, 128), "fp16")
        b = _rand((1, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_nonpow2_n_fp16(self):
        """T043: non-power-2 N"""
        a = _rand((1, 1, 128), "fp16")
        b = _rand((1, 128, 3072), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_decode_scorev_worst_fp16(self):
        """T044: decode score x V -- worst"""
        a = _rand((1, 1, 2048), "fp16")
        b = _rand((1, 2048, 128), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_bug_tsp4032_underutil_fp16(self):
        """T045: THE BUG -- tsp#4032"""
        a = _rand((1, 1, 4096), "fp16")
        b = _rand((1, 4096, 25600), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_reference_full_util_fp16(self):
        """T046: reference (N%2048=0)"""
        a = _rand((1, 1, 4096), "fp16")
        b = _rand((1, 4096, 26624), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_at_span_limit_fp16(self):
        """T047: at span limit"""
        a = _rand((1, 1, 4096), "fp16")
        b = _rand((1, 4096, 32768), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_decode_qkT_bf16(self):
        """T048: bf16 decode QK^T"""
        a = _rand((1, 1, 128), "bf16")
        b = _rand((1, 128, 2048), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_m1_bug_tsp4032_underutil_bf16(self):
        """T049: bf16 underutil case"""
        a = _rand((1, 1, 4096), "bf16")
        b = _rand((1, 4096, 25600), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestBmm3DCostModelPass2(_WDTestCase):
    """3D bmm, B=1 M>1 -> all gates pass -> Pass 2's cost model actually
    runs and picks a split."""

    def test_bmm_3d_b1_mgt1_speculative_decode_underfill_fp16(self):
        """T050: M underfill -- speculative decode"""
        a = _rand((1, 4, 128), "fp16")
        b = _rand((1, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_mgt1_m16_underfill_boundary_fp16(self):
        """T051: M=16 underfill boundary"""
        a = _rand((1, 16, 128), "fp16")
        b = _rand((1, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_mgt1_prefill_qkT_standard_fp16(self):
        """T052: standard m-split -- prefill QK^T"""
        a = _rand((1, 2048, 128), "fp16")
        b = _rand((1, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_mgt1_scorev_k_gt_n_penalty_fp16(self):
        """T053: K>>N shape penalty -- score x V"""
        a = _rand((1, 2048, 2048), "fp16")
        b = _rand((1, 2048, 128), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_mgt1_mlp_upproj_wide_n_fp16(self):
        """T054: wide-N penalty -- MLP up-proj"""
        a = _rand((1, 2048, 4096), "fp16")
        b = _rand((1, 4096, 11008), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_mgt1_prefill_qkT_standard_bf16(self):
        """T055: bf16 prefill QK^T"""
        a = _rand((1, 2048, 128), "bf16")
        b = _rand((1, 128, 2048), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_3d_b1_mgt1_span_limit_plus1_elem_fp16(self):
        """T056: 1 elem over span limit"""
        a = _rand((1, 2048, 4096), "fp16")
        b = _rand((1, 4096, 32832), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_bmm_3d_b1_mgt1_prefill_qkT_fp32_expect_fail(self):
        """T057: fp32 prefill QK^T"""
        a = _rand((1, 2048, 128), "fp32")
        b = _rand((1, 128, 2048), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_bmm_3d_b1_mgt1_scorev_fp32_expect_fail(self):
        """T058: fp32 score x V"""
        a = _rand((1, 2048, 2048), "fp32")
        b = _rand((1, 2048, 128), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_bmm_3d_b1_mgt1_mlp_upproj_fp32_expect_fail(self):
        """T059: fp32 MLP up-proj"""
        a = _rand((1, 2048, 4096), "fp32")
        b = _rand((1, 4096, 11008), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestBmm4DGreedyPass3(_WDTestCase):
    """4D bmm, B>1 M=1 -> Gate 3 fires again -> Pass 3's b x N split."""

    def test_bmm_4d_bgt1_m1_batch2_fp16(self):
        """T070: B=2"""
        a = _rand((2, 1, 128), "fp16")
        b = _rand((2, 128, 1024), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_m1_batch4_fp16(self):
        """T071: B=4"""
        a = _rand((4, 1, 128), "fp16")
        b = _rand((4, 128, 512), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_m1_batch8_fp16(self):
        """T072: B=8"""
        a = _rand((8, 1, 128), "fp16")
        b = _rand((8, 128, 256), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_m1_batch16_fp16(self):
        """T073: B=16"""
        a = _rand((16, 1, 128), "fp16")
        b = _rand((16, 128, 128), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_m1_batch32_fp16(self):
        """T074: B=32"""
        a = _rand((32, 1, 64), "fp16")
        b = _rand((32, 64, 64), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_m1_odd_batch5_fp16(self):
        """T075: odd B=5"""
        a = _rand((5, 1, 128), "fp16")
        b = _rand((5, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_m1_odd_batch7_fp16(self):
        """T076: odd B=7"""
        a = _rand((7, 1, 128), "fp16")
        b = _rand((7, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestBmm4DCostModelPass2(_WDTestCase):
    """4D bmm, B>1 M>1 -> Pass 2 cost model with the batch-split penalty."""

    def test_bmm_4d_bgt1_mgt1_multihead_prefill_qkT_fp16(self):
        """T080: multi-head prefill QK^T"""
        a = _rand((4, 2048, 128), "fp16")
        b = _rand((4, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_mgt1_batched_heads_qkT_fp16(self):
        """T081: batched heads QK^T"""
        a = _rand((4, 2048, 512), "fp16")
        b = _rand((4, 512, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_mgt1_bert_style_batched_bf16(self):
        """T082: BERT-style batched"""
        a = _rand((8, 512, 128), "bf16")
        b = _rand((8, 128, 512), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_4d_bgt1_mgt1_batched_prefill_bf16(self):
        """T083: bf16 batched prefill"""
        a = _rand((4, 2048, 128), "bf16")
        b = _rand((4, 128, 2048), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_bmm_4d_bgt1_mgt1_multihead_prefill_fp32_expect_fail(self):
        """T084: fp32 multi-head prefill"""
        a = _rand((4, 2048, 128), "fp32")
        b = _rand((4, 128, 2048), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_bmm_4d_bgt1_mgt1_batched_heads_fp32_expect_fail(self):
        """T085: fp32 batched heads"""
        a = _rand((4, 2048, 512), "fp32")
        b = _rand((4, 512, 2048), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestBmmRankLimit5D6D(_WDTestCase):
    """5D/6D bmm -- rank>4 batched matmul. Behavior has flip-flopped across
    different runs (rejected in one, compiling fine in another); treated
    as a normal case here since the most recently confirmed runs show it
    compiling successfully."""

    def test_bmm_5d_gqa_motivated_rank_limit_fp16(self):
        """T120: 5D GQA-motivated -- confirmed compiles fine"""
        a = _rand((2, 2, 4, 256, 256), "fp16")
        b = _rand((2, 2, 4, 256, 256), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_5d_gqa_motivated_rank_limit_bf16(self):
        """T121: 5D GQA-motivated bf16 -- confirmed compiles fine"""
        a = _rand((2, 2, 4, 256, 256), "bf16")
        b = _rand((2, 2, 4, 256, 256), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_6d_deeper_nesting_rank_limit_fp16(self):
        """T130: 6D deeper nesting -- confirmed compiles fine"""
        a = _rand((2, 2, 2, 2, 256, 256), "fp16")
        b = _rand((2, 2, 2, 2, 256, 256), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_6d_deeper_nesting_rank_limit_bf16(self):
        """T131: 6D deeper nesting bf16 -- confirmed compiles fine"""
        a = _rand((2, 2, 2, 2, 256, 256), "bf16")
        b = _rand((2, 2, 2, 2, 256, 256), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestRealModelAttentionMLP(_WDTestCase):
    """Real-model attention QK^T and MLP up-proj shapes (Llama, gpt-oss,
    Mistral, granite) across every dtype that keeps the tensor under the
    256 MiB span limit."""

    def test_bmm_realmodel_llama31_8b_attn_qkT_fp16(self):
        """T140: Llama-3.1-8B attn QK^T"""
        a = _rand((1, 2048, 128), "fp16")
        b = _rand((1, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_realmodel_gptoss20b_attn_qkT_fp16(self):
        """T141: gpt-oss-20b attn QK^T"""
        a = _rand((1, 2048, 64), "fp16")
        b = _rand((1, 64, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_realmodel_mistral_small24b_attn_qkT_fp16(self):
        """T142: Mistral-Small-24B attn QK^T"""
        a = _rand((1, 2048, 128), "fp16")
        b = _rand((1, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_bmm_realmodel_granite3x_8b_attn_qkT_fp16(self):
        """T143: granite-3.x-8b attn QK^T"""
        a = _rand((1, 2048, 128), "fp16")
        b = _rand((1, 128, 2048), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_mm_realmodel_llama_mlp_upproj_fp32_expect_fail(self):
        """T144: Llama MLP up-proj fp32"""
        a = _rand((2048, 4096), "fp32")
        b = _rand((4096, 14336), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_realmodel_llama_mlp_upproj_fp16(self):
        """T145: Llama MLP up-proj fp16"""
        a = _rand((2048, 4096), "fp16")
        b = _rand((4096, 14336), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_realmodel_llama_mlp_upproj_bf16(self):
        """T146: Llama MLP up-proj bf16"""
        a = _rand((2048, 4096), "bf16")
        b = _rand((4096, 14336), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #4310: raw-cast FP8 (no quantize_fp8_with_scale metadata) is unsupported by the FP8 matmul lowering"
    )
    def test_mm_realmodel_llama_mlp_upproj_fp8_expect_fail(self):
        """T147: Llama MLP up-proj fp8"""
        a = _rand((2048, 4096), "fp8")
        b = _rand((4096, 14336), "fp8")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_mm_realmodel_gptoss_perexpert_fp32_expect_fail(self):
        """T148: gpt-oss per-expert fp32"""
        a = _rand((2048, 2880), "fp32")
        b = _rand((2880, 2880), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_realmodel_gptoss_perexpert_fp16(self):
        """T149: gpt-oss per-expert fp16"""
        a = _rand((2048, 2880), "fp16")
        b = _rand((2880, 2880), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_realmodel_gptoss_perexpert_bf16(self):
        """T14A: gpt-oss per-expert bf16"""
        a = _rand((2048, 2880), "bf16")
        b = _rand((2880, 2880), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #4310: raw-cast FP8 (no quantize_fp8_with_scale metadata) is unsupported by the FP8 matmul lowering"
    )
    def test_mm_realmodel_gptoss_perexpert_fp8_expect_fail(self):
        """T14B: gpt-oss per-expert fp8"""
        a = _rand((2048, 2880), "fp8")
        b = _rand((2880, 2880), "fp8")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #4310: raw-cast FP8 (no quantize_fp8_with_scale metadata) is unsupported by the FP8 matmul lowering"
    )
    def test_mm_realmodel_mistral_mlp_upproj_fp8_expect_fail(self):
        """T14C: Mistral MLP up-proj fp8"""
        a = _rand((2048, 5120), "fp8")
        b = _rand((5120, 32768), "fp8")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #1794: batchmatmul not yet in SPYRE_FP32_OPS -- Inductor raises Unsupported: matmul on DataFormats.IEEE_FP32"
    )
    def test_mm_realmodel_granite_mlp_upproj_fp32_expect_fail(self):
        """T14D: granite MLP up-proj fp32"""
        a = _rand((2048, 4096), "fp32")
        b = _rand((4096, 12800), "fp32")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_realmodel_granite_mlp_upproj_fp16(self):
        """T14E: granite MLP up-proj fp16"""
        a = _rand((2048, 4096), "fp16")
        b = _rand((4096, 12800), "fp16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    def test_mm_realmodel_granite_mlp_upproj_bf16(self):
        """T14F: granite MLP up-proj bf16"""
        a = _rand((2048, 4096), "bf16")
        b = _rand((4096, 12800), "bf16")
        torch.compile(torch.matmul, dynamic=False)(a, b)

    @pytest.mark.xfail(
        reason="Issue #4310: raw-cast FP8 (no quantize_fp8_with_scale metadata) is unsupported by the FP8 matmul lowering"
    )
    def test_mm_realmodel_granite_mlp_upproj_fp8_expect_fail(self):
        """T14G: granite MLP up-proj fp8"""
        a = _rand((2048, 4096), "fp8")
        b = _rand((4096, 12800), "fp8")
        torch.compile(torch.matmul, dynamic=False)(a, b)


class TestPointwise1D(_WDTestCase):
    """1D pointwise & reduction -- always Pass 3 (Gate 1 always fails)."""

    def test_pointwise_1d_add_31_idle_fp16(self):
        """T001: 31 idle expected"""
        x = _rand((64,), "fp16")
        y = _rand((64,), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_1d_add_full_util_fp16(self):
        """T002: full util expected"""
        x = _rand((2048,), "fp16")
        y = _rand((2048,), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_1d_add_full_util_large_fp16(self):
        """T003: full util expected"""
        x = _rand((4096,), "fp16")
        y = _rand((4096,), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_1d_add_20_idle_bf16(self):
        """T004: 20 idle expected"""
        x = _rand((768,), "bf16")
        y = _rand((768,), "bf16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_1d_mul_verify_cores_fp16(self):
        """T005: verify core count"""
        x = _rand((11008,), "fp16")
        y = _rand((11008,), "fp16")
        torch.compile(lambda a, b: a * b, dynamic=False)(x, y)

    def test_reduction_1d_mean_no_split_possible_fp16(self):
        """T006: no split possible"""
        x = _rand((2048,), "fp16")
        torch.compile(lambda t: torch.mean(t, dim=0), dynamic=False)(x)

    def test_softmax_1d_reduce_over_only_dim_fp16(self):
        """T007: reduce over only dim"""
        x = _rand((2048,), "fp16")
        torch.compile(lambda t: torch.softmax(t, dim=0), dynamic=False)(x)


class TestPointwise2D(_WDTestCase):
    """2D pointwise & reduction, including the layernorm/softmax cases
    whose reduction dim must never be split."""

    def test_pointwise_2d_add_prefill_residual_fp16(self):
        """T020: prefill residual add"""
        x = _rand((2048, 4096), "fp16")
        y = _rand((2048, 4096), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_2d_add_decode_residual_fp16(self):
        """T021: decode residual add"""
        x = _rand((1, 4096), "fp16")
        y = _rand((1, 4096), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_2d_add_prefill_residual_bf16(self):
        """T022: bf16 prefill residual"""
        x = _rand((2048, 4096), "bf16")
        y = _rand((2048, 4096), "bf16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_2d_mul_swiglu_gate_fp16(self):
        """T023: SwiGLU gate 2-D"""
        x = _rand((2048, 11008), "fp16")
        y = _rand((2048, 11008), "fp16")
        torch.compile(lambda a, b: a * b, dynamic=False)(x, y)

    def test_pointwise_2d_add_large_p1_may_split_bf16(self):
        """T024: large, P1 may split"""
        x = _rand((8192, 4096), "bf16")
        y = _rand((8192, 4096), "bf16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_layernorm_2d_granite_llama_hidden_fp16(self):
        """T030: Granite/Llama hidden"""
        x = _rand((2048, 4096), "fp16")
        normalized_shape = (4096,)
        weight = _rand(normalized_shape, "fp16")
        bias = _rand(normalized_shape, "fp16")

        def fn(t, w, b):
            return F.layer_norm(t, normalized_shape, w, b)

        torch.compile(fn, dynamic=False)(x, weight, bias)

    def test_layernorm_2d_decode_fp16(self):
        """T031: decode layernorm"""
        x = _rand((1, 4096), "fp16")
        normalized_shape = (4096,)
        weight = _rand(normalized_shape, "fp16")
        bias = _rand(normalized_shape, "fp16")

        def fn(t, w, b):
            return F.layer_norm(t, normalized_shape, w, b)

        torch.compile(fn, dynamic=False)(x, weight, bias)

    def test_layernorm_2d_bert_hidden_bf16(self):
        """T032: BERT hidden"""
        x = _rand((49152, 768), "bf16")
        normalized_shape = (768,)
        weight = _rand(normalized_shape, "bf16")
        bias = _rand(normalized_shape, "bf16")

        def fn(t, w, b):
            return F.layer_norm(t, normalized_shape, w, b)

        torch.compile(fn, dynamic=False)(x, weight, bias)

    def test_softmax_2d_attention_scores_fp16(self):
        """T033: attention scores 2-D"""
        x = _rand((2048, 2048), "fp16")
        torch.compile(lambda t: torch.softmax(t, dim=-1), dynamic=False)(x)

    def test_reduction_2d_mean_global_avgpool_bf16(self):
        """T034: global avg pool"""
        x = _rand((32, 768), "bf16")
        torch.compile(lambda t: torch.mean(t, dim=1), dynamic=False)(x)


class TestPointwise3D(_WDTestCase):
    """3D pointwise, including real-model MLP activation shapes."""

    def test_pointwise_3d_add_prefill_residual_fp16(self):
        """T060: prefill residual 3-D"""
        x = _rand((1, 2048, 4096), "fp16")
        y = _rand((1, 2048, 4096), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_3d_add_decode_residual_fp16(self):
        """T061: decode residual 3-D"""
        x = _rand((1, 1, 4096), "fp16")
        y = _rand((1, 1, 4096), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_3d_add_prefill_residual_bf16(self):
        """T062: bf16 prefill residual 3-D"""
        x = _rand((1, 2048, 4096), "bf16")
        y = _rand((1, 2048, 4096), "bf16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_activation_3d_silu_llama_mlp_bf16(self):
        """T063: Llama MLP activation"""
        x = _rand((1, 2048, 14336), "bf16")
        torch.compile(F.silu, dynamic=False)(x)

    def test_activation_3d_gelu_granite_mlp_bf16(self):
        """T064: Granite MLP activation"""
        x = _rand((1, 2048, 16384), "bf16")
        torch.compile(F.gelu, dynamic=False)(x)

    def test_pointwise_3d_mul_swiglu_gate_fp16(self):
        """T065: SwiGLU gate 3-D"""
        x = _rand((1, 2048, 11008), "fp16")
        y = _rand((1, 2048, 11008), "fp16")
        torch.compile(lambda a, b: a * b, dynamic=False)(x, y)

    def test_pointwise_3d_add_large_p1_may_split_bf16(self):
        """T066: large, P1 may split"""
        x = _rand((8, 4096, 4096), "bf16")
        y = _rand((8, 4096, 4096), "bf16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_activation_3d_silu_gptoss_mlp_bf16(self):
        """T067: gpt-oss MLP activation"""
        x = _rand((1, 2048, 2880), "bf16")
        torch.compile(F.silu, dynamic=False)(x)

    def test_activation_3d_silu_mistral_mlp_bf16(self):
        """T068: Mistral MLP activation"""
        x = _rand((1, 2048, 32768), "bf16")
        torch.compile(F.silu, dynamic=False)(x)

    def test_activation_3d_gelu_granite_mlp_real_intermediate_bf16(self):
        """T069: granite MLP activation"""
        x = _rand((1, 2048, 12800), "bf16")
        torch.compile(F.gelu, dynamic=False)(x)


class TestPointwise4D(_WDTestCase):
    """4D pointwise & reduction -- attention masks, RoPE, decode/prefill
    softmax."""

    def test_pointwise_4d_add_decode_attn_mask_fp16(self):
        """T090: decode attn mask add"""
        x = _rand((1, 32, 1, 2048), "fp16")
        y = _rand((1, 32, 1, 2048), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_4d_add_prefill_attn_mask_large_fp16(self):
        """T091: prefill attn mask (large)"""
        x = _rand((1, 32, 2048, 2048), "fp16")
        y = _rand((1, 32, 2048, 2048), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_4d_mul_rope_elementwise_fp16(self):
        """T092: RoPE elementwise"""
        x = _rand((1, 32, 2048, 128), "fp16")
        y = _rand((1, 32, 2048, 128), "fp16")
        torch.compile(lambda a, b: a * b, dynamic=False)(x, y)

    def test_pointwise_4d_add_batched_large_bf16(self):
        """T093: batched large add"""
        x = _rand((1, 32, 2048, 4096), "bf16")
        y = _rand((1, 32, 2048, 4096), "bf16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_softmax_4d_attention_scores_prefill_fp16(self):
        """T095: attention scores prefill"""
        x = _rand((1, 32, 2048, 2048), "fp16")
        torch.compile(lambda t: torch.softmax(t, dim=-1), dynamic=False)(x)

    def test_softmax_4d_decode_attention_scores_fp16(self):
        """T096: decode attention scores"""
        x = _rand((1, 32, 1, 2048), "fp16")
        torch.compile(lambda t: torch.softmax(t, dim=-1), dynamic=False)(x)

    def test_layernorm_4d_headwise_bf16(self):
        """T097: head-wise layernorm"""
        x = _rand((1, 32, 2048, 128), "bf16")
        normalized_shape = (128,)
        weight = _rand(normalized_shape, "bf16")
        bias = _rand(normalized_shape, "bf16")

        def fn(t, w, b):
            return F.layer_norm(t, normalized_shape, w, b)

        torch.compile(fn, dynamic=False)(x, weight, bias)


class TestPointwise5D6D(_WDTestCase):
    """5D/6D pointwise output -- grouped-head and deeply batched shapes."""

    def test_pointwise_5d_add_grouped_head_residual_fp16(self):
        """T100: grouped-head residual"""
        x = _rand((1, 2, 16, 2048, 128), "fp16")
        y = _rand((1, 2, 16, 2048, 128), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_5d_mul_gqa_elementwise_gate_bf16(self):
        """T101: GQA elementwise gate"""
        x = _rand((1, 4, 8, 2048, 64), "bf16")
        y = _rand((1, 4, 8, 2048, 64), "bf16")
        torch.compile(lambda a, b: a * b, dynamic=False)(x, y)

    def test_softmax_5d_grouped_head_attn_scores_fp16(self):
        """T102: grouped-head attn scores"""
        x = _rand((1, 2, 16, 2048, 2048), "fp16")
        torch.compile(lambda t: torch.softmax(t, dim=-1), dynamic=False)(x)

    def test_pointwise_6d_add_deeply_batched_residual_fp16(self):
        """T103: deeply batched residual"""
        x = _rand((1, 2, 4, 8, 128, 64), "fp16")
        y = _rand((1, 2, 4, 8, 128, 64), "fp16")
        torch.compile(lambda a, b: a + b, dynamic=False)(x, y)

    def test_pointwise_6d_mul_deeply_batched_gate_bf16(self):
        """T104: deeply batched gate"""
        x = _rand((1, 2, 4, 8, 32, 64), "bf16")
        y = _rand((1, 2, 4, 8, 32, 64), "bf16")
        torch.compile(lambda a, b: a * b, dynamic=False)(x, y)


class TestHandoffCheck(unittest.TestCase):
    """Planner -> apply_splits -> scheduler handoff, mirroring
    work_division_handoff_check.py's own snapshot-compare structure
    (cost_model_planner -> apply_splits -> scheduler_transport, then
    assert_valid + assert_same on each pair of stages) across several
    different shapes, ops, and corruption patterns."""

    def test_handoff_preserves_split_across_stages(self):
        """The exact case work_division_handoff_check.py runs by default."""
        sizes = {"M": 2048, "N": 128, "K": 2048}
        planner = cost_model_planner({"M": 16, "N": 1, "K": 2})
        committed = apply_splits(planner)
        received = dict(committed)  # scheduler_transport, no bug

        assert_valid(committed, sizes)
        assert_valid(received, sizes)
        assert_same("planner", planner, "after apply_splits", committed, sizes)
        assert_same(
            "after apply_splits", committed, "scheduler received", received, sizes
        )
        self.assertEqual(prod(received.values()), 32)

    def test_handoff_simulate_bug_is_detected(self):
        """work_division_handoff_check.py --simulate-bug: K=2 is
        incorrectly converted into N=2 during scheduler transport."""
        sizes = {"M": 2048, "N": 128, "K": 2048}
        planner = cost_model_planner({"M": 16, "N": 1, "K": 2})
        committed = apply_splits(planner)
        received = dict(committed)
        received["N"] = 2
        received["K"] = 1

        assert_valid(committed, sizes)
        assert_valid(received, sizes)
        assert_same("planner", planner, "after apply_splits", committed, sizes)
        with self.assertRaises(AssertionError):
            assert_same(
                "after apply_splits", committed, "scheduler received", received, sizes
            )

    def test_handoff_preserves_explicit_unsplit_dimension(self):
        """A dimension explicitly committed as split=1 (not just missing
        from the dict) must survive the handoff exactly as 1."""
        sizes = {"M": 32, "N": 2048, "K": 64}
        planner = cost_model_planner({"M": 1, "N": 32, "K": 1})
        committed = apply_splits(planner)
        received = dict(committed)

        assert_valid(committed, sizes)
        assert_valid(received, sizes)
        assert_same("planner", planner, "after apply_splits", committed, sizes)
        assert_same(
            "after apply_splits", committed, "scheduler received", received, sizes
        )
        self.assertEqual(received["M"], 1)
        self.assertEqual(received["K"], 1)

    def test_handoff_preserves_pointwise_style_split(self):
        """The handoff logic shouldn't care whether a plan came from Pass 2
        (M/N/K-named keys) or Pass 3's greedy splitter (stick-based d0/d1
        keys) -- confirms it explicitly rather than assuming it."""
        sizes = {"d0": 2048, "d1": 4096}
        planner = cost_model_planner({"d0": 32, "d1": 1})
        committed = apply_splits(planner)
        received = dict(committed)

        assert_valid(committed, sizes)
        assert_valid(received, sizes)
        assert_same("planner", planner, "after apply_splits", committed, sizes)
        assert_same(
            "after apply_splits", committed, "scheduler received", received, sizes
        )
        self.assertEqual(received, {"d0": 32, "d1": 1})

    def test_handoff_preserves_softmax_style_split(self):
        """softmax's reduction dim (d1) must stay committed at split=1
        through every handoff stage -- if it silently gained a split
        during transport, each core would normalize only part of a row."""
        sizes = {"d0": 2048, "d1": 2048}
        planner = cost_model_planner({"d0": 32, "d1": 1})
        committed = apply_splits(planner)
        received = dict(committed)

        assert_valid(committed, sizes)
        assert_valid(received, sizes)
        assert_same("planner", planner, "after apply_splits", committed, sizes)
        assert_same(
            "after apply_splits", committed, "scheduler received", received, sizes
        )
        self.assertEqual(received["d1"], 1)

    def test_handoff_preserves_decode_layernorm_no_parallelism(self):
        """The M=1 decode edge case: every dim committed at split=1 (no
        parallelism available, 1 core used) must round-trip just as
        cleanly as a fully-utilized plan."""
        sizes = {"d0": 1, "d1": 4096}
        planner = cost_model_planner({"d0": 1, "d1": 1})
        committed = apply_splits(planner)
        received = dict(committed)

        assert_valid(committed, sizes)
        assert_valid(received, sizes)
        assert_same("planner", planner, "after apply_splits", committed, sizes)
        assert_same(
            "after apply_splits", committed, "scheduler received", received, sizes
        )
        self.assertEqual(prod(received.values()), 1)

    def test_handoff_preserves_vocab_width_bug_shape(self):
        """Ties the handoff check directly to the tracked tsp#4032 shape
        (N=25600 -> 25 cores, not 32). The 25-core outcome is Pass 3's
        cost to bear (a separate, already tracked issue); the handoff's
        job is just not to lose the plan in transit."""
        sizes = {"M": 1, "N": 25600, "K": 4096}
        planner = cost_model_planner({"M": 1, "N": 25, "K": 1})
        committed = apply_splits(planner)
        received = dict(committed)

        assert_valid(committed, sizes)
        assert_valid(received, sizes)
        assert_same("planner", planner, "after apply_splits", committed, sizes)
        assert_same(
            "after apply_splits", committed, "scheduler received", received, sizes
        )
        self.assertEqual(prod(received.values()), 25)

    def test_handoff_preserves_4d_split_with_batch_dimension(self):
        """A plan with more than 3 keys (B/M/N/K, the 4D batched-bmm
        shape) must survive the handoff just as completely as the
        original 3-key M/N/K case."""
        sizes = {"B": 4, "M": 2048, "N": 2048, "K": 128}
        planner = cost_model_planner({"B": 4, "M": 8, "N": 1, "K": 1})
        committed = apply_splits(planner)
        received = dict(committed)

        assert_valid(committed, sizes)
        assert_valid(received, sizes)
        assert_same("planner", planner, "after apply_splits", committed, sizes)
        assert_same(
            "after apply_splits", committed, "scheduler received", received, sizes
        )
        self.assertEqual(received, {"B": 4, "M": 8, "N": 1, "K": 1})
        self.assertLessEqual(prod(received.values()), MAX_CORES)

    def test_handoff_detects_dropped_dimension_key(self):
        """A different real bug shape than relabeling: a key vanishing
        entirely between apply_splits and the scheduler, defaulting to 1
        via .get(dim, 1). Must be caught exactly like a relabeled key."""
        sizes = {"M": 2048, "N": 128, "K": 2048}
        planner = cost_model_planner({"M": 16, "N": 1, "K": 2})
        committed = apply_splits(planner)
        received = dict(committed)
        del received["K"]  # dropped entirely, not relabeled

        assert_valid(committed, sizes)
        assert_same("planner", planner, "after apply_splits", committed, sizes)
        with self.assertRaises(AssertionError):
            assert_same(
                "after apply_splits", committed, "scheduler received", received, sizes
            )

    def test_handoff_detects_magnitude_corruption(self):
        """A corruption that keeps the same dimension name but changes the
        split's magnitude (K:2 silently becoming K:1). Deliberately HALVES
        rather than doubles K: doubling would push cores from 32 to 64,
        tripping assert_valid's core-budget check first and masking the
        comparison this test means to exercise."""
        sizes = {"M": 2048, "N": 128, "K": 2048}
        planner = cost_model_planner({"M": 16, "N": 1, "K": 2})
        committed = apply_splits(planner)
        received = dict(committed)
        received["K"] = 1

        assert_valid(committed, sizes)
        assert_valid(received, sizes)
        assert_same("planner", planner, "after apply_splits", committed, sizes)
        with self.assertRaises(AssertionError):
            assert_same(
                "after apply_splits", committed, "scheduler received", received, sizes
            )

    def test_handoff_detects_bug_introduced_before_apply_splits(self):
        """Everything else here corrupts apply_splits -> scheduler. This
        one corrupts planner -> apply_splits instead, proving the FIRST
        assert_same call actually catches a bug, not just the second."""
        sizes = {"M": 2048, "N": 128, "K": 2048}
        planner = cost_model_planner({"M": 16, "N": 1, "K": 2})
        committed = apply_splits(planner)
        committed["M"] = 1  # corrupted at commit time, before the scheduler

        with self.assertRaises(AssertionError):
            assert_same("planner", planner, "after apply_splits", committed, sizes)

    def test_handoff_rejects_split_exceeding_max_cores(self):
        """assert_valid has only ever been exercised by legal input so
        far. Confirms it actually rejects a plan whose core product
        exceeds MAX_CORES."""
        sizes = {"M": 2048, "N": 128, "K": 2048}
        planner = cost_model_planner({"M": 64, "N": 1, "K": 2})  # 128 cores
        committed = apply_splits(planner)

        with self.assertRaises(AssertionError):
            assert_valid(committed, sizes)

    def test_handoff_rejects_split_not_dividing_dimension_size(self):
        """Likewise, confirms assert_valid rejects a split that is
        core-legal but doesn't evenly divide its dimension's real size."""
        sizes = {"M": 100, "N": 128, "K": 2048}  # 100 is not divisible by 3
        planner = cost_model_planner({"M": 3, "N": 1, "K": 1})
        committed = apply_splits(planner)

        with self.assertRaises(AssertionError):
            assert_valid(committed, sizes)


if __name__ == "__main__":
    unittest.main()
