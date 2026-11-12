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

"""Model-derived FVT for FP8 scaled batched matmul (``spyre.scaled_bmm``).

Sections 1-10, 13 and 14 of the FP8 scaled_bmm FVT plan; Sections 11 and 12 are in
tests/inductor/test_fp8_operations.py and tests/inductor/test_fp8_prequantized_load.py.
Shapes come from Granite-3.3/4.1-8B-FP8 and Gemma-4-26B-A4B. Test ids are the FVT
case keys plus ``-cores<N>``.

Operands are signed and exactly representable in FP8 E4M3, so device quantization
has nothing to round and a mis-laid-out operand cannot sum to the right value.
Cases that fail today are listed in _KNOWN_FAILURES and run as strict xfails.
"""

import contextlib
import logging

import regex as re

import pytest
import torch
from torch._dynamo.exc import TorchRuntimeError
from torch._inductor.exc import InductorError
from utils_inductor import (
    DEVICE,
    _compile_and_run,
    cached_randn,
    expects_raise,
    strict_xfail,
)

from torch_spyre._inductor import config
from torch_spyre._inductor.constants import FP8_E4M3FN_MAX
from torch_spyre._inductor.logging_utils import get_inductor_logger

FP8 = torch.float8_e4m3fn
FP8_MIN_NORMAL = 2.0**-6
FP8_MIN_SUBNORMAL = 2.0**-9
# Per-element bound: ACC_ULPS x unit roundoff x sum|a*b|. Measured error is at most
# 4.7 unit roundoffs; a mis-laid-out operand gives 57-266. A flat atol/rtol cannot
# separate the two.
ACC_ULPS = 16
# fp16 subnormal spacing, for outputs a small scale pushes toward 0.
FP16_OUTPUT_FLOOR = 4 * 2.0**-24

# Every case runs at 1 and 32 cores; cases whose split depends on a dim add one even
# and one odd count.
CORES_BASE = (1, 32)


SHAPE_CASES = [
    ("SB3D-01", "3d_qk_32x64x128x64", (32, 64, 128), (32, 128, 64), CORES_BASE),
    ("SB3D-02", "3d_pv_32x64x64x128", (32, 64, 64), (32, 64, 128), CORES_BASE),
    ("SB3D-03", "3d_qk_32x128x128x128", (32, 128, 128), (32, 128, 128), CORES_BASE),
    ("SB3D-04", "3d_qk_64x128x128x128", (64, 128, 128), (64, 128, 128), CORES_BASE),
    ("SB3D-05", "3d_qk_32x512x128x512", (32, 512, 128), (32, 128, 512), CORES_BASE),
    ("SB3D-06", "3d_pv_32x512x512x128", (32, 512, 512), (32, 512, 128), CORES_BASE),
    (
        "SB3D-07",
        "3d_decode_qk_32x1x128x2048",
        (32, 1, 128),
        (32, 128, 2048),
        (1, 7, 8, 32),
    ),
    (
        "SB3D-08",
        "3d_moe_bmm_as_captured_8x2816x704x1",
        (8, 2816, 704),
        (8, 704, 1),
        (1, 4, 11, 32),
    ),
    (
        "SB3D-09",
        "3d_moe_bmm_weight_as_mat2_8x1x704x2816",
        (8, 1, 704),
        (8, 704, 2816),
        (1, 8, 11, 32),
    ),
    (
        "SB3D-10",
        "3d_moe_bmm_as_captured_8x1408x2816x1",
        (8, 1408, 2816),
        (8, 2816, 1),
        (1, 11, 16, 32),
    ),
    (
        "SB3D-11",
        "3d_moe_bmm_weight_as_mat2_8x1x2816x1408",
        (8, 1, 2816),
        (8, 2816, 1408),
        (1, 4, 11, 32),
    ),
    (
        "SB3D-12",
        "3d_moe_bmm_as_captured_96x2816x704x1",
        (96, 2816, 704),
        (96, 704, 1),
        (1, 3, 11, 32),
    ),
    (
        "SB3D-13",
        "3d_moe_bmm_weight_as_mat2_96x1x704x2816",
        (96, 1, 704),
        (96, 704, 2816),
        (1, 3, 16, 32),
    ),
    (
        "SB3D-14",
        "3d_moe_bmm_as_captured_96x1408x2816x1",
        (96, 1408, 2816),
        (96, 2816, 1),
        (1, 7, 11, 32),
    ),
    (
        "SB3D-15",
        "3d_moe_bmm_weight_as_mat2_96x1x2816x1408",
        (96, 1, 2816),
        (96, 2816, 1408),
        (1, 3, 8, 32),
    ),
    (
        "SBQK-01",
        "4d_prefill_qk_1x32x1024x128x1024",
        (1, 32, 1024, 128),
        (1, 32, 128, 1024),
        CORES_BASE,
    ),
    (
        "SBQK-02",
        "4d_prefill_qk_1x32x2048x128x2048",
        (1, 32, 2048, 128),
        (1, 32, 128, 2048),
        CORES_BASE,
    ),
    (
        "SBQK-03",
        "4d_prefill_qk_2x32x128x128x128",
        (2, 32, 128, 128),
        (2, 32, 128, 128),
        CORES_BASE,
    ),
    (
        "SBQK-04",
        "4d_prefill_qk_2x32x512x128x512",
        (2, 32, 512, 128),
        (2, 32, 128, 512),
        CORES_BASE,
    ),
    (
        "SBQK-05",
        "4d_prefill_qk_4x32x128x128x128",
        (4, 32, 128, 128),
        (4, 32, 128, 128),
        CORES_BASE,
    ),
    (
        "SBQK-06",
        "4d_prefill_qk_4x32x512x128x512",
        (4, 32, 512, 128),
        (4, 32, 128, 512),
        CORES_BASE,
    ),
    (
        "SBPV-01",
        "4d_prefill_pv_1x32x64x64x128",
        (1, 32, 64, 64),
        (1, 32, 64, 128),
        CORES_BASE,
    ),
    (
        "SBPV-02",
        "4d_prefill_pv_1x32x256x256x128",
        (1, 32, 256, 256),
        (1, 32, 256, 128),
        CORES_BASE,
    ),
    (
        "SBPV-03",
        "4d_prefill_pv_1x32x512x512x128",
        (1, 32, 512, 512),
        (1, 32, 512, 128),
        CORES_BASE,
    ),
    (
        "SBPV-04",
        "4d_prefill_pv_1x32x1024x1024x128",
        (1, 32, 1024, 1024),
        (1, 32, 1024, 128),
        CORES_BASE,
    ),
    (
        "SBPV-05",
        "4d_prefill_pv_1x32x2048x2048x128",
        (1, 32, 2048, 2048),
        (1, 32, 2048, 128),
        CORES_BASE,
    ),
    (
        "SBPV-06",
        "4d_prefill_pv_2x32x512x512x128",
        (2, 32, 512, 512),
        (2, 32, 512, 128),
        CORES_BASE,
    ),
    (
        "SBPV-07",
        "4d_prefill_pv_4x32x512x512x128",
        (4, 32, 512, 512),
        (4, 32, 512, 128),
        CORES_BASE,
    ),
    (
        "SBDEC-01",
        "4d_decode_qk_1x32x1x128x64",
        (1, 32, 1, 128),
        (1, 32, 128, 64),
        (1, 3, 8, 32),
    ),
    (
        "SBDEC-02",
        "4d_decode_pv_1x32x1x64x128",
        (1, 32, 1, 64),
        (1, 32, 64, 128),
        (1, 7, 16, 32),
    ),
    (
        "SBDEC-03",
        "4d_decode_qk_1x32x1x128x512",
        (1, 32, 1, 128),
        (1, 32, 128, 512),
        (1, 4, 7, 32),
    ),
    (
        "SBDEC-04",
        "4d_decode_pv_1x32x1x512x128",
        (1, 32, 1, 512),
        (1, 32, 512, 128),
        (1, 3, 16, 32),
    ),
    (
        "SBDEC-05",
        "4d_decode_qk_1x32x1x128x2048",
        (1, 32, 1, 128),
        (1, 32, 128, 2048),
        (1, 7, 8, 32),
    ),
    (
        "SBDEC-06",
        "4d_decode_pv_1x32x1x2048x128",
        (1, 32, 1, 2048),
        (1, 32, 2048, 128),
        (1, 3, 4, 32),
    ),
    (
        "SBDEC-07",
        "4d_decode_qk_4x32x1x128x64",
        (4, 32, 1, 128),
        (4, 32, 128, 64),
        (1, 4, 7, 32),
    ),
    (
        "SBDEC-08",
        "4d_decode_pv_4x32x1x64x128",
        (4, 32, 1, 64),
        (4, 32, 64, 128),
        (1, 3, 16, 32),
    ),
    (
        "SBDEC-09",
        "4d_decode_qk_4x32x1x128x512",
        (4, 32, 1, 128),
        (4, 32, 128, 512),
        (1, 7, 16, 32),
    ),
    (
        "SBDEC-10",
        "4d_decode_pv_4x32x1x512x128",
        (4, 32, 1, 512),
        (4, 32, 512, 128),
        (1, 3, 4, 32),
    ),
    (
        "SBDEC-11",
        "4d_decode_qk_4x32x1x128x2048",
        (4, 32, 1, 128),
        (4, 32, 128, 2048),
        (1, 7, 8, 32),
    ),
    (
        "SBDEC-12",
        "4d_decode_pv_4x32x1x2048x128",
        (4, 32, 1, 2048),
        (4, 32, 2048, 128),
        (1, 3, 8, 32),
    ),
    (
        "SBDEC-13",
        "4d_global_decode_qk_1x16x1x512x2048",
        (1, 16, 1, 512),
        (1, 16, 512, 2048),
        (1, 7, 16, 32),
    ),
    (
        "SBDEC-14",
        "4d_global_decode_pv_1x16x1x2048x512",
        (1, 16, 1, 2048),
        (1, 16, 2048, 512),
        (1, 3, 16, 32),
    ),
    (
        "SBDEC-15",
        "4d_sliding_decode_qk_1x16x1x256x1024",
        (1, 16, 1, 256),
        (1, 16, 256, 1024),
        (1, 7, 8, 32),
    ),
    (
        "SBDEC-16",
        "4d_sliding_decode_pv_1x16x1x1024x256",
        (1, 16, 1, 1024),
        (1, 16, 1024, 256),
        (1, 3, 4, 32),
    ),
    (
        "SBMDL-01",
        "4d_granite-3.3-8b-instruct-FP8_prefill_qk_1x32x41x128x2048",
        (1, 32, 41, 128),
        (1, 32, 128, 2048),
        CORES_BASE,
    ),
    (
        "SBMDL-02",
        "4d_granite-3.3-8b-instruct-FP8_prefill_pv_1x32x41x2048x128",
        (1, 32, 41, 2048),
        (1, 32, 2048, 128),
        CORES_BASE,
    ),
    (
        "SBMDL-03",
        "4d_granite-4.1-8b-fp8_prefill_qk_1x32x29x128x2048",
        (1, 32, 29, 128),
        (1, 32, 128, 2048),
        CORES_BASE,
    ),
    (
        "SBMDL-04",
        "4d_granite-4.1-8b-fp8_prefill_pv_1x32x29x2048x128",
        (1, 32, 29, 2048),
        (1, 32, 2048, 128),
        CORES_BASE,
    ),
    (
        "SBMDL-05",
        "4d_global_prefill_qk_1x16x12x512x12",
        (1, 16, 12, 512),
        (1, 16, 512, 12),
        (1, 3, 16, 32),
    ),
    (
        "SBMDL-06",
        "4d_global_prefill_pv_1x16x12x12x512",
        (1, 16, 12, 12),
        (1, 16, 12, 512),
        (1, 4, 7, 32),
    ),
    (
        "SBMDL-07",
        "4d_sliding_prefill_qk_1x16x12x256x1024",
        (1, 16, 12, 256),
        (1, 16, 256, 1024),
        (1, 3, 8, 32),
    ),
    (
        "SBMDL-08",
        "4d_sliding_prefill_pv_1x16x12x1024x256",
        (1, 16, 12, 1024),
        (1, 16, 1024, 256),
        (1, 7, 16, 32),
    ),
    (
        "SBMDL-09",
        "4d_sliding_prefill_qk_1x16x24x256x1024",
        (1, 16, 24, 256),
        (1, 16, 256, 1024),
        (1, 3, 4, 32),
    ),
    (
        "SBMDL-10",
        "4d_sliding_prefill_pv_1x16x24x1024x256",
        (1, 16, 24, 1024),
        (1, 16, 1024, 256),
        (1, 7, 8, 32),
    ),
    (
        "SBMDL-11",
        "4d_global_prefill_qk_1x16x24x512x24",
        (1, 16, 24, 512),
        (1, 16, 512, 24),
        (1, 3, 16, 32),
    ),
    (
        "SBMDL-12",
        "4d_global_prefill_pv_1x16x24x24x512",
        (1, 16, 24, 24),
        (1, 16, 24, 512),
        (1, 7, 8, 32),
    ),
]

# fvt_id -> (key, mat1, mat2, core counts)
_CASES = {c[0]: c[1:] for c in SHAPE_CASES}


@contextlib.contextmanager
def _cores(n):
    """Compile for ``n`` cores with static shapes.

    HBM pool planning is off: every compiled program keeps its reservation (#3979),
    and this module compiles hundreds of them.
    """
    with (
        config.patch({"sencores": n, "hbm_pool_planning": False}),
        torch._dynamo.config.patch({"automatic_dynamic_shapes": False}),
    ):
        yield


def _fp8_exact(shape, tag):
    """Signed values exactly representable in FP8 E4M3, as fp32 (no subnormals)."""
    x = cached_randn(shape, differentiation=tag).float()
    x = torch.where(x.abs() < FP8_MIN_NORMAL, torch.zeros_like(x), x)
    return x.clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX).to(FP8).float()


def _sign(shape, tag):
    s = cached_randn(shape, differentiation=(tag, "sign"))
    return torch.where(s < 0, -1.0, 1.0)


def _scaled_bmm_fn(out_dtype=torch.float16, use_fast_accum=False):
    """Quantize both operands on the device, then scaled_bmm."""

    def fn(a, b, qa_s, qb_s, sa, sb):
        q_a = torch.ops.spyre.quantize_fp8_with_scale(a, qa_s)
        q_b = torch.ops.spyre.quantize_weight_fp8_with_scale(b, qb_s)
        return torch.ops.spyre.scaled_bmm(
            q_a,
            q_b,
            scale_a=sa,
            scale_b=sb,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )

    return fn


def _assert_within_bound(out, ref, absum, out_dtype=torch.float16):
    """|out - ref| <= ACC_ULPS * u(out_dtype) * sum|ab| + floor, element-wise."""
    u = max(torch.finfo(out_dtype).eps, torch.finfo(torch.float16).eps) / 2
    bound = ACC_ULPS * u * absum + FP16_OUTPUT_FLOOR
    err = (out.float() - ref).abs()
    bad = ~(err <= bound)
    assert out.shape == ref.shape, f"shape {tuple(out.shape)} != {tuple(ref.shape)}"
    assert not bad.any(), (
        f"{bad.sum().item()} / {bad.numel()} elements outside "
        f"{ACC_ULPS} x u x sum|ab|; worst err/sum|ab| = "
        f"{(err / absum.clamp(min=1e-30)).max().item():.5f} "
        f"(bound {ACC_ULPS * u:.5f})"
    )


def _run_exact(
    fvt_id,
    sencores,
    scale_a=1.0,
    scale_b=1.0,
    scale_dtype=torch.float16,
    out_dtype=torch.float16,
    scale_b_dtype=None,
):
    """Run scaled_bmm on FP8-exact inputs; return (out, CPU result, sum|ab|).

    A ``None`` scale is omitted from scaled_bmm; its operand is quantized with 1.0.
    """
    _, mat1, mat2, _ = _CASES[fvt_id]
    ra = _fp8_exact(mat1, (fvt_id, "a"))
    rb = _fp8_exact(mat2, (fvt_id, "b"))
    sa_v = 1.0 if scale_a is None else scale_a
    sb_v = 1.0 if scale_b is None else scale_b
    a = (ra * sa_v).to(torch.float16)
    b = (rb * sb_v).to(torch.float16)
    qa_s = torch.tensor(sa_v, dtype=torch.float16)
    qb_s = torch.tensor(sb_v, dtype=torch.float16)
    sa = None if scale_a is None else torch.tensor(sa_v, dtype=scale_dtype)
    sb_dtype = scale_dtype if scale_b_dtype is None else scale_b_dtype
    sb = None if scale_b is None else torch.tensor(sb_v, dtype=sb_dtype)
    with _cores(sencores):
        out = _compile_and_run(
            _scaled_bmm_fn(out_dtype=out_dtype), (a, b, qa_s, qb_s, sa, sb), DEVICE
        )
    s = sa_v * sb_v
    ref = (ra @ rb) * s
    absum = (ra.abs() @ rb.abs()) * abs(s)
    return out, ref, absum


def _params(fvt_ids, *extra):
    """pytest params for each case x its core counts."""
    tags = [str(e) for e in extra]
    return [
        pytest.param(
            fvt_id, *extra, n, id="-".join([_CASES[fvt_id][0], *tags, f"cores{n}"])
        )
        for fvt_id in fvt_ids
        for n in _CASES[fvt_id][3]
    ]


_KNOWN_FAILURES = [
    (
        "test_shapes",
        r"3d_moe_bmm_as_captured_(8|96)x2816x704x1-cores\d+",
        InductorError,
        "#6076: [fp8] scaled_bmm: N=1 MoE expert matmul fails layout planning "
        "when K is not a stick multiple",
    ),
    (
        "test_shapes",
        r"3d_moe_bmm_as_captured_96x1408x2816x1-cores1"
        r"|3d_moe_bmm_weight_as_mat2_96x1x(2816x1408|704x2816)-cores\d+",
        InductorError,
        "#6082: [fp8] scaled_bmm: per-core span limit rejects MoE shapes without "
        "splitting the batch dimension",
    ),
    (
        "test_shapes",
        r"3d_moe_bmm_as_captured_8x1408x2816x1-cores1"
        r"|3d_moe_bmm_as_captured_96x1408x2816x1-cores11",
        AssertionError,
        "#5924: [fp8] batchmatmulfp8 returns wrong values when N is a multiple of"
        " 64 but not of 128 (K aligned)",
    ),
    (
        "test_shapes",
        r"3d_moe_bmm_weight_as_mat2_8x1x704x2816-cores\d+",
        InductorError,
        "#6079: [fp8] scaled_bmm: dbo-opt DtException when the contraction K is "
        "not a multiple of 128",
    ),
    (
        "test_shapes",
        r"(3d_pv_32x64x64x128|4d_prefill_pv_1x32x64x64x128"
        r"|4d_decode_pv_(1|4)x32x1x64x128)-cores\d+",
        InductorError,
        '#6080: [fp8] scaled_bmm: P@V with K=64 rejected "cannot restickify any '
        'input layout of x to carry x_var=d3"',
    ),
    (
        "test_shapes",
        r"4d_global_prefill_pv_1x16x(12x12|24x24)x512-cores\d+",
        InductorError,
        "#5524: [test_full_fp8_pipeline] 5 model-derived shapes with unaligned K "
        "(2112) fail to compile",
    ),
    (
        "test_scales",
        r"(?!.*-None-None-).*-torch\.float32-cores\d+",
        AssertionError,
        "#6061: fp16 tensor multiplied by an fp32 scalar tensor returns wrong values",
    ),
    (
        "test_scale_dtypes",
        r".*",
        TorchRuntimeError,
        "#6053: [fp8] scaled_bmm: out_dtype=None is rejected by the op schema",
    ),
    (
        "test_boundary",
        r".*-tiny-.*",
        AssertionError,
        "#6067: [fp8] FP8 subnormals mis-decoded in the matmul; #6084: [fp8] "
        "quantize_fp8_with_scale corrupts E4M3 subnormal values",
    ),
    (
        "test_out_dtype",
        r".*-None-cores\d+",
        TorchRuntimeError,
        "#6053: [fp8] scaled_bmm: out_dtype=None is rejected by the op schema",
    ),
    (
        "test_rejections",
        r"mat[12]_fp16-cores\d+",
        AssertionError,
        "#6203: [fp8] scaled_bmm: FP8 operand guard in lower_bmm_fp8 is bypassed "
        "when one operand is fp16",
    ),
    (
        "test_padded_inputs",
        r"pad_n_3d_moe_bmm_as_captured_8x2816x704x1-cores\d+",
        InductorError,
        "#6079: [fp8] scaled_bmm: dbo-opt DtException when the contraction K is "
        "not a multiple of 128",
    ),
    (
        "test_for_each_tile",
        r".*",
        InductorError,
        "#5423: `batchmatmulfp8` fails layout planning inside `for_each_tile`: "
        '"no mechanism to resolve stick incompatibility"',
    ),
]


def _expect_known_failure(request):
    """Strict xfail for a test listed in _KNOWN_FAILURES."""
    test, case = request.node.originalname, request.node.callspec.id
    for name, pattern, raises, reason in _KNOWN_FAILURES:
        if name == test and re.fullmatch(pattern, case):
            strict_xfail(request, raises=raises, reason=reason)
            return


class TestScaledBmmShapes:
    """Sections 1-5: FR-003 (3D) and FR-004 (4D) at model-derived shapes."""

    @pytest.mark.parametrize("fvt_id,sencores", _params(_CASES))
    def test_shapes(self, fvt_id, sencores, request):
        _expect_known_failure(request)
        _assert_within_bound(*_run_exact(fvt_id, sencores))


_GRANITE_SCALE_PAIRS = [
    (1.0, 1.0),
    (2.0, 1.0),
    (1.0, 3.0),
    (2.0, 3.0),
    (0.5, 1.0),
    (1.0, 0.25),
    (0.5, 0.25),
    (0.125, 8.0),
    (None, None),
    (2.0, None),
    (None, 3.0),
]
_GEMMA_SCALE_PAIRS = [(1.0, 1.0), (2.0, 3.0), (0.5, 0.25), (None, 3.0)]
_SCALE_GROUPS = [
    (("SB3D-03", "SBQK-03"), _GRANITE_SCALE_PAIRS),
    (("SB3D-11", "SBDEC-13"), _GEMMA_SCALE_PAIRS),
]


_SCALE_DTYPE_PAIRS = [
    (torch.float16, torch.float16),
    (torch.float32, torch.float32),
    (torch.float32, torch.float16),
    (torch.float16, torch.float32),
]


# A small scale is paired with a larger one
_GRANITE_SCALE_VALUES = [
    (0.0, 1.5),
    (2.5, 0.0),
    (-1.5, 2.0),
    (1.25, -0.75),
    (-2.5, -1.75),
    (0.0001, 2.5),
    (3.75, 0.0001),
    (0.00001, 4.5),
    (5.0, 0.00001),
    (1.5, 2.25),
    (4.9, 3.3),
    (5.0, 5.0),
]
_GEMMA_SCALE_VALUES = [
    (0.0, 1.5),
    (-1.5, 2.0),
    (0.0001, 2.5),
    (0.00001, 4.5),
    (4.9, 3.3),
]
_SCALE_VALUE_GROUPS = [
    (("SB3D-03", "SBQK-03"), _GRANITE_SCALE_VALUES),
    (("SB3D-11", "SBDEC-13"), _GEMMA_SCALE_VALUES),
]


def _run_scale_value(fvt_id, sencores, scale_a, scale_b):
    """Like _run_exact, with any fp16 scale value.

    A zero scale cannot quantize, so that operand is quantized with 1.0.
    """
    _, mat1, mat2, _ = _CASES[fvt_id]
    ra = _fp8_exact(mat1, (fvt_id, "a"))
    rb = _fp8_exact(mat2, (fvt_id, "b"))
    qa_v = scale_a if scale_a != 0 else 1.0
    qb_v = scale_b if scale_b != 0 else 1.0
    a = (ra * qa_v).to(torch.float16)
    b = (rb * qb_v).to(torch.float16)
    qa_s = torch.tensor(qa_v, dtype=torch.float16)
    qb_s = torch.tensor(qb_v, dtype=torch.float16)
    sa = torch.tensor(scale_a, dtype=torch.float16)
    sb = torch.tensor(scale_b, dtype=torch.float16)
    with _cores(sencores):
        out = _compile_and_run(_scaled_bmm_fn(), (a, b, qa_s, qb_s, sa, sb), DEVICE)
    q_a = (a.float() / qa_s.float()).clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX)
    q_b = (b.float() / qb_s.float()).clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX)
    q_a, q_b = q_a.to(FP8).float(), q_b.to(FP8).float()
    s = sa.float().item() * sb.float().item()
    return out, (q_a @ q_b) * s, (q_a.abs() @ q_b.abs()) * abs(s)


def _scale_params():
    return [
        p
        for fvt_ids, pairs in _SCALE_GROUPS
        for sa, sb in pairs
        for dt in (torch.float16, torch.float32)
        for p in _params(fvt_ids, sa, sb, dt)
    ]


class TestScaledBmmScales:
    """Section 6: scale_a / scale_b are applied inside the op (FR-005, 6.3)."""

    @pytest.mark.parametrize(
        "fvt_id,scale_a,scale_b,scale_dtype,sencores", _scale_params()
    )
    def test_scales(self, fvt_id, scale_a, scale_b, scale_dtype, sencores, request):
        _expect_known_failure(request)
        _assert_within_bound(
            *_run_exact(fvt_id, sencores, scale_a, scale_b, scale_dtype=scale_dtype)
        )

    @pytest.mark.parametrize(
        "fvt_id,scale_a,scale_b,sencores",
        [
            p
            for fvt_ids, values in _SCALE_VALUE_GROUPS
            for sa, sb in values
            for p in _params(fvt_ids, sa, sb)
        ],
    )
    def test_scale_values(self, fvt_id, scale_a, scale_b, sencores):
        """Scales of 0, negative, ~1e-4, 1e-5 and decimals up to 5 (FR-005)."""
        out, ref, absum = _run_scale_value(fvt_id, sencores, scale_a, scale_b)
        assert torch.isfinite(out).all(), "non-finite output"
        _assert_within_bound(out, ref, absum)

    @pytest.mark.parametrize(
        "fvt_id,scale_a_dtype,scale_b_dtype,sencores",
        [
            p
            for fvt_ids, _ in _SCALE_GROUPS
            for dta, dtb in _SCALE_DTYPE_PAIRS
            for p in _params(fvt_ids, dta, dtb)
        ],
    )
    def test_scale_dtypes(
        self, fvt_id, scale_a_dtype, scale_b_dtype, sencores, request
    ):
        """Scale dtype pairs with out_dtype left at its default (None)."""
        _expect_known_failure(request)
        out, ref, absum = _run_exact(
            fvt_id,
            sencores,
            2.0,
            3.0,
            scale_dtype=scale_a_dtype,
            scale_b_dtype=scale_b_dtype,
            out_dtype=None,
        )
        assert out.dtype == torch.float16, f"default out_dtype gave {out.dtype}"
        _assert_within_bound(out, ref, absum)


# The one section where quantization rounds or clamps: mat1 near, at, beyond or
# below the E4M3 range.
_BOUNDARY_GROUPS = [
    ("SBQK-03", ("near_max", "at_max", "saturate", "tiny"), [(1.0, 1.0), (2.0, 3.0)]),
    ("SB3D-11", ("saturate", "tiny"), [(1.0, 1.0)]),
]


def _boundary_params():
    return [
        p
        for fvt_id, modes, pairs in _BOUNDARY_GROUPS
        for mode in modes
        for sa, sb in pairs
        for p in _params([fvt_id], mode, sa, sb)
    ]


class TestScaledBmmBoundary:
    """Section 7: values near, at, beyond and below the E4M3 range."""

    @pytest.mark.parametrize("fvt_id,mode,scale_a,scale_b,sencores", _boundary_params())
    def test_boundary(self, fvt_id, mode, scale_a, scale_b, sencores, request):
        _expect_known_failure(request)
        _, mat1, mat2, _ = _CASES[fvt_id]
        u = cached_randn(mat1, differentiation=(fvt_id, mode), abs=True).float()
        u = u.clamp(max=3.0) / 3.0
        if mode == "near_max":
            mag = 440.0 + 8.0 * u
        elif mode == "at_max":
            mag = torch.full(mat1, FP8_E4M3FN_MAX)
        elif mode == "saturate":
            mag = 449.0 + 1551.0 * u
        else:  # "tiny": uniform(2^-9, 2^-6), the E4M3 subnormal range
            mag = FP8_MIN_SUBNORMAL + (FP8_MIN_NORMAL - FP8_MIN_SUBNORMAL) * u
        a = (mag * _sign(mat1, (fvt_id, mode, "a")) * scale_a).to(torch.float16)
        other = FP8_E4M3FN_MAX if mode == "tiny" else 0.125
        rb = other * _sign(mat2, (fvt_id, mode, "b"))
        b = (rb * scale_b).to(torch.float16)
        sa = torch.tensor(scale_a, dtype=torch.float16)
        sb = torch.tensor(scale_b, dtype=torch.float16)
        with _cores(sencores):
            out = _compile_and_run(_scaled_bmm_fn(), (a, b, sa, sb, sa, sb), DEVICE)

        ra = (a.float() / scale_a).clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX)
        ra = ra.to(FP8).float()
        s = scale_a * scale_b
        assert torch.isfinite(out).all(), "non-finite output"
        _assert_within_bound(out, (ra @ rb) * s, (ra.abs() @ rb.abs()) * s)


_OUT_DTYPE_GROUPS = [
    ("SBQK-03", (None, torch.float16)),
    ("SB3D-11", (None, torch.float16)),
]

_UNSUPPORTED_OUT_DTYPES = (torch.bfloat16, torch.float32, torch.float8_e4m3fn)

_OUT_DTYPE_NAMES = {
    torch.bfloat16: ("bfloat16",),
    torch.float32: ("float32", "IEEE_FP32"),
    torch.float8_e4m3fn: ("float8_e4m3fn", "SEN143_FP8"),
}


class TestScaledBmmParameters:
    """Section 8: out_dtype resolution and the ignored use_fast_accum flag."""

    @pytest.mark.parametrize(
        "fvt_id,out_dtype,sencores",
        [
            p
            for fvt_id, dtypes in _OUT_DTYPE_GROUPS
            for dt in dtypes
            for p in _params([fvt_id], dt)
        ],
    )
    def test_out_dtype(self, fvt_id, out_dtype, sencores, request):
        """out_dtype=None and out_dtype=torch.float16 both give fp16 (FRS 5.1)."""
        _expect_known_failure(request)
        expected = out_dtype or torch.float16
        out, ref, absum = _run_exact(fvt_id, sencores, out_dtype=out_dtype)
        assert out.dtype == expected, f"out_dtype={out_dtype} gave {out.dtype}"
        _assert_within_bound(out, ref, absum, expected)

    @pytest.mark.parametrize(
        "fvt_id,use_fast_accum,sencores",
        [
            p
            for fvt_id in ("SBQK-03", "SB3D-11")
            for ufa in (True, False)
            for p in _params([fvt_id], ufa)
        ],
    )
    def test_use_fast_accum_warns(self, fvt_id, use_fast_accum, sencores, caplog):
        """use_fast_accum only warns: logged when True, silent when False."""
        _, mat1, mat2, _ = _CASES[fvt_id]
        a = _fp8_exact(mat1, (fvt_id, "a")).to(torch.float16)
        b = _fp8_exact(mat2, (fvt_id, "b")).to(torch.float16)
        one = torch.tensor(1.0, dtype=torch.float16)
        args = (a, b, one, one, one, one)
        logger = get_inductor_logger("decompositions")

        with (
            _cores(sencores),
            torch._functorch.config.patch({"enable_autograd_cache": False}),
            caplog.at_level(logging.WARNING, logger=logger.name),
        ):
            fn = _scaled_bmm_fn(use_fast_accum=use_fast_accum)
            _compile_and_run(fn, args, DEVICE)
        warned = any("use_fast_accum" in r.getMessage() for r in caplog.records)
        assert warned is use_fast_accum, (
            f"use_fast_accum={use_fast_accum}: "
            f"{'expected' if use_fast_accum else 'did not expect'} a warning"
        )


_G4D = (1, 32, 128, 128)
# (key, mat1, mat2, mat1 dtype, mat2 dtype, message fragment)
REJECTION_CASES = [
    (
        "mat1_fp16",
        _G4D,
        _G4D,
        torch.float16,
        FP8,
        "Expected FP8 input for mat1, got torch.float16",
    ),
    (
        "mat2_fp16",
        _G4D,
        _G4D,
        FP8,
        torch.float16,
        "Expected FP8 input for mat2, got torch.float16",
    ),
    ("rank2", (128, 128), (128, 128), FP8, FP8, "not supported"),
    ("rank5", (1, 1, 32, 128, 128), (1, 1, 32, 128, 128), FP8, FP8, "not supported"),
    ("rank_mismatch_4_3", _G4D, (32, 128, 128), FP8, FP8, "not supported"),
    ("batch_mismatch", (2, 32, 128, 128), _G4D, FP8, FP8, "not supported"),
    ("gqa_heads_unexpanded", _G4D, (1, 8, 128, 128), FP8, FP8, "not supported"),
    ("broadcast_batch", (2, 32, 128, 128), (1, 1, 128, 128), FP8, FP8, "not supported"),
    ("k_mismatch", _G4D, (1, 32, 256, 128), FP8, FP8, "not supported"),
    (
        "gqa_unexpanded_global",
        (1, 16, 64, 512),
        (1, 2, 512, 2048),
        FP8,
        FP8,
        "not supported",
    ),
    (
        "gqa_unexpanded_sliding",
        (1, 16, 64, 256),
        (1, 8, 256, 1024),
        FP8,
        FP8,
        "not supported",
    ),
]


class TestScaledBmmRejections:
    """Section 9: invalid inputs raise instead of returning a tensor (9.1)."""

    @expects_raise
    @pytest.mark.parametrize(
        "key,mat1,mat2,dt1,dt2,fragment,sencores",
        [
            pytest.param(*c, n, id=f"{c[0]}-cores{n}")
            for c in REJECTION_CASES
            for n in CORES_BASE
        ],
    )
    def test_rejections(self, key, mat1, mat2, dt1, dt2, fragment, sencores, request):
        _expect_known_failure(request)
        a = cached_randn(mat1, differentiation=(key, "mat1")).to(dt1)
        b = cached_randn(mat2, differentiation=(key, "mat2")).to(dt2)

        def fn(a, b):
            return torch.ops.spyre.scaled_bmm(a, b, out_dtype=torch.float16)

        with _cores(sencores), pytest.raises(InductorError, match=fragment):
            _compile_and_run(fn, (a, b), DEVICE)

    @expects_raise
    @pytest.mark.parametrize(
        "fvt_id,out_dtype,sencores",
        [p for dt in _UNSUPPORTED_OUT_DTYPES for p in _params(["SBQK-03"], dt)],
    )
    def test_rejects_out_dtype(self, fvt_id, out_dtype, sencores, request):
        """A non-FP16 out_dtype raises an error naming the dtype (SBREJ-12..14).

        bfloat16 may instead return FP16-equivalent output within the bound.
        """
        if out_dtype == torch.bfloat16:
            strict_xfail(
                request,
                raises=AssertionError,
                reason="#6061: fp16 tensor multiplied by an fp32 scalar tensor returns"
                " wrong values",
            )
            try:
                out, ref, absum = _run_exact(fvt_id, sencores, out_dtype=out_dtype)
            except InductorError as e:  # a rejection is one of the two passes
                _assert_names_out_dtype(e, out_dtype)
                return
            assert out.dtype in (torch.float16, torch.bfloat16), (
                f"out_dtype={out_dtype} returned {out.dtype}"
            )
            _assert_within_bound(out, ref, absum, out.dtype)
            return
        with pytest.raises(InductorError) as excinfo:
            _run_exact(fvt_id, sencores, out_dtype=out_dtype)
        _assert_names_out_dtype(excinfo.value, out_dtype)


def _assert_names_out_dtype(error, out_dtype):
    """The error names the refused out_dtype or its device format."""
    msg = str(error)
    names = _OUT_DTYPE_NAMES[out_dtype]
    assert any(n.lower() in msg.lower() for n in names), (
        f"error for out_dtype={out_dtype} names none of {names}: {msg[:300]}"
    )


# (heads, head_dim, core counts)
_VIEW_GEOMETRIES = [(32, 128, CORES_BASE), (16, 256, (1, 7, 16, 32))]
_VIEW_SEQ = 128
_VIEW_VARIANTS = ("head_split_q", "transposed_k", "permuted_q")


def _view_inputs(variant, heads, head_dim):
    """FP8-exact (ra, rb) as views of a [1, S, H*D] activation and a K cache."""
    s = _VIEW_SEQ
    x = _fp8_exact((1, s, heads * head_dim), "views_x")
    k = _fp8_exact((1, heads, s, head_dim), "views_k")
    q4 = x.view(1, s, heads, head_dim)
    if variant == "head_split_q":
        return q4.transpose(1, 2), k.transpose(-1, -2).contiguous()
    if variant == "transposed_k":
        return q4.transpose(1, 2).contiguous(), k.transpose(-1, -2)
    return q4.permute(0, 2, 1, 3), k.transpose(-1, -2).contiguous()


class TestScaledBmmNonContiguous:
    """Section 10: views from an attention block give the contiguous result."""

    @pytest.mark.parametrize(
        "heads,head_dim,sencores",
        [
            pytest.param(h, d, n, id=f"{h}x{d}-cores{n}")
            for h, d, cores in _VIEW_GEOMETRIES
            for n in cores
        ],
    )
    def test_non_contiguous_views(self, heads, head_dim, sencores):
        """Each view operand matches the CPU result and its contiguous copy."""
        one = torch.tensor(1.0, dtype=torch.float16)
        fn = _scaled_bmm_fn()
        failures = []
        for variant in _VIEW_VARIANTS:
            ra, rb = _view_inputs(variant, heads, head_dim)
            a, b = ra.to(torch.float16), rb.to(torch.float16)  # keeps the strides
            assert not (a.is_contiguous() and b.is_contiguous())
            try:
                with _cores(sencores):
                    out = _compile_and_run(fn, (a, b, one, one, one, one), DEVICE)
                    ctrl = _compile_and_run(
                        fn, (a.contiguous(), b.contiguous(), one, one, one, one), DEVICE
                    )
                _assert_within_bound(out, ra @ rb, ra.abs() @ rb.abs())
                assert torch.equal(out, ctrl), "view result differs from contiguous"
            except Exception as e:  # noqa: BLE001 - report every sub-case
                failures.append(f"{variant}: {type(e).__name__}: {e}"[:400])
        assert not failures, "\n".join(failures)


def _run_fullgraph(fn, args):
    """Compile with fullgraph=True, which for_each_tile needs, and run on Spyre.

    The full Dynamo reset stops a later compile reusing a wrong-shape artifact.
    """
    torch._dynamo.reset()
    torch._inductor.codecache.FxGraphCache.clear()
    dev = torch.device(DEVICE)
    dev_args = [a.to(dev) if isinstance(a, torch.Tensor) else a for a in args]
    compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False)
    return compiled(*dev_args).cpu()


# (fvt id, test key, B, M, D, page N, cores)
_FET_PAGED = [
    ("SBFT-01", "paged_qk_granite_decode_32x1x128", 32, 1, 128, 128, (1, 32)),
    ("SBFT-02", "paged_qk_granite_prefill_32x512x128", 32, 512, 128, 128, (1, 32)),
    ("SBFT-03", "paged_pv_granite_32x1x128", 32, 1, 128, 128, (1, 32)),
    ("SBFT-04", "paged_qk_gemma_global_16x1x512", 16, 1, 512, 128, (1, 7, 16, 32)),
    ("SBFT-05", "paged_qk_gemma_sliding_16x1x256", 16, 1, 256, 128, (1, 3, 8, 32)),
]


def _fet_params():
    out = [
        pytest.param(fvt_id, key, b, m, d, n, c, id=f"{key}-cores{c}")
        for fvt_id, key, b, m, d, n, cores in _FET_PAGED
        for c in cores
    ]
    out += [
        pytest.param(
            "SBFT-06",
            "map_experts_gemma_moe_8x1x2816x1408",
            8,
            1,
            2816,
            1408,
            c,
            id=f"map_experts_gemma_moe_8x1x2816x1408-cores{c}",
        )
        for c in (1, 4, 11, 32)
    ]
    return out


def _fet_block(act, weight_2d):
    """The FP8 block: per-token scale, quantize, scaled_bmm, rescale."""
    s = torch.ops.spyre.quantscalepertokenfp8(act, 1.0)
    q = torch.ops.spyre.quantize_fp8_with_scale(act, s)
    w = torch.ops.spyre.qfp8wt(weight_2d)
    return torch.ops.spyre.scaled_bmm(q, w, out_dtype=torch.float16) * s


_FET_AMAX = 4.0
_FET_MIN = 2.0**-4


def _fet_act(shape, tag):
    """FP8-exact activation whose per-token amax is exactly ``_FET_AMAX``."""
    x = _fp8_exact(shape, tag)
    mag = x.abs().clamp(_FET_MIN, _FET_AMAX)
    x = torch.where(x < 0, -mag, mag)
    x[..., 0] = _FET_AMAX
    return x.to(torch.float16)


def _fet_ref(act, tiles, pv, moe):
    """CPU reference for the whole loop, and its abs-sum for the bound."""
    q = act.float() / _FET_AMAX
    if moe:
        w = tiles.float().transpose(-2, -1)
        return (q @ w) * _FET_AMAX, (q.abs() @ w.abs()) * _FET_AMAX
    ref = absum = None
    for i in range(tiles.shape[0]):
        w = tiles[i].float()
        if not pv:
            w = w.transpose(-2, -1)
        o, a = (q @ w) * _FET_AMAX, (q.abs() @ w.abs()) * _FET_AMAX
        ref = o if ref is None else ref + o
        absum = a if absum is None else absum + a
    return ref, absum


class TestScaledBmmForEachTile:
    """Section 14: the FP8 attention block inside a for_each_tile body."""

    @pytest.mark.parametrize("fvt_id,key,b,m,d,n,sencores", _fet_params())
    def test_for_each_tile(self, fvt_id, key, b, m, d, n, sencores, request):
        """The FP8 block inside for_each_tile gives the CPU result."""
        _expect_known_failure(request)
        from torch_spyre._inductor.wsr import for_each_tile

        pages = 4
        moe = fvt_id == "SBFT-06"
        if moe:  # map over experts, weight is [B, K, N]
            x = _fet_act((b, m, d), (fvt_id, "x"))
            w = _fp8_exact((b, n, d), (fvt_id, "w")).to(torch.float16)
            tile = 2

            def expert(xt, wt):
                return _fet_block(xt, wt.transpose(-2, -1).contiguous())

            def tiled(x, w):
                def body(_carry, tiles):
                    xt, wt = tiles
                    return None, expert(xt, wt)

                _, out = for_each_tile(
                    body, (x, w), dims=(0, 0), tile_size=tile, out_dim=0
                )
                return out

            args = (x, w)
            tiles_t, pv = w, False
        else:
            act_s = n if fvt_id == "SBFT-03" else d
            pv = fvt_id == "SBFT-03"
            act = _fet_act((b, m, act_s), (fvt_id, "act"))
            pg = _fp8_exact((pages, b, n, d), (fvt_id, "pg")).to(torch.float16)
            acc0 = torch.zeros(b, m, n, dtype=torch.float16)

            def one_page(a, page):
                w2d = page if pv else page.transpose(-2, -1).contiguous()
                return _fet_block(a, w2d)

            def tiled(act, pg, acc0):
                def body(carry, tiles):
                    page, a, _ = tiles
                    (acc,) = carry
                    return (acc + one_page(a, page.squeeze(0)),), None

                (out,), _ = for_each_tile(
                    body,
                    (pg, act, acc0),
                    dims=(0, None, None),
                    tile_size=1,
                    init=(acc0,),
                )
                return out

            args = (act, pg, acc0)
            tiles_t = pg

        with _cores(sencores):
            out = _run_fullgraph(tiled, args)
        ref, absum = _fet_ref(args[0], tiles_t, pv, moe)
        _assert_within_bound(out, ref, absum)


# Zero padding must not change the result; padded rows and columns must be zero.
# (FVT id, test key, source case, axis, padded extent, core counts)
_PAD_CASES = [
    ("SBPAD-01", "pad_k_4d_prefill_pv_1x32x64x64x128", "SBPV-01", "k", 128, CORES_BASE),
    (
        "SBPAD-02",
        "pad_k_4d_global_prefill_pv_1x16x12x12x512",
        "SBMDL-06",
        "k",
        128,
        (1, 4, 7, 32),
    ),
    (
        "SBPAD-03",
        "pad_k_3d_moe_bmm_weight_as_mat2_8x1x704x2816",
        "SB3D-09",
        "k",
        768,
        (1, 8, 11, 32),
    ),
    (
        "SBPAD-04",
        "pad_n_3d_moe_bmm_as_captured_8x2816x704x1",
        "SB3D-08",
        "n",
        64,
        (1, 4, 11, 32),
    ),
    (
        "SBPAD-05",
        "pad_n_4d_global_prefill_qk_1x16x12x512x12",
        "SBMDL-05",
        "n",
        64,
        (1, 3, 16, 32),
    ),
    (
        "SBPAD-06",
        "pad_m_4d_granite-3.3-8b-instruct-FP8_prefill_qk_1x32x41x128x2048",
        "SBMDL-01",
        "m",
        64,
        CORES_BASE,
    ),
]


class TestScaledBmmPadded:
    """Section 13: zero-padded operands give the unpadded result, pad region zero."""

    @pytest.mark.parametrize(
        "fvt_id,key,source,axis,extent,sencores",
        [
            pytest.param(f, k, s, ax, e, c, id=f"{k}-cores{c}")
            for f, k, s, ax, e, cores in _PAD_CASES
            for c in cores
        ],
    )
    def test_padded_inputs(self, fvt_id, key, source, axis, extent, sencores, request):
        """The live part matches the unpadded CPU result; the padding is zero."""
        _expect_known_failure(request)
        _, mat1, mat2, _ = _CASES[source]
        ra = _fp8_exact(mat1, (source, "a"))
        rb = _fp8_exact(mat2, (source, "b"))
        m, k, n = mat1[-2], mat1[-1], mat2[-1]

        pa, pb = ra, rb
        if axis == "k":
            pa = torch.nn.functional.pad(ra, (0, extent - k))
            pb = torch.nn.functional.pad(rb, (0, 0, 0, extent - k))
        elif axis == "n":
            pb = torch.nn.functional.pad(rb, (0, extent - n))
        else:
            pa = torch.nn.functional.pad(ra, (0, 0, 0, extent - m))

        one = torch.tensor(1.0, dtype=torch.float16)
        args = (pa.to(torch.float16), pb.to(torch.float16), one, one, one, one)
        with _cores(sencores):
            out = _compile_and_run(_scaled_bmm_fn(), args, DEVICE)

        ref = ra @ rb
        absum = ra.abs() @ rb.abs()
        live, pad = out, None
        if axis == "n":
            live, pad = out[..., :n], out[..., n:]
        elif axis == "m":
            live, pad = out[..., :m, :], out[..., m:, :]
        _assert_within_bound(live, ref, absum)
        if pad is not None:
            assert torch.equal(pad.float(), torch.zeros_like(pad.float())), (
                f"{(pad != 0).sum().item()} / {pad.numel()} padded elements "
                f"are non-zero (max |value| {pad.abs().max().item()})"
            )
