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

"""Shared cases and helpers for the FP8 scaled_bmm FVT.

The tests are split across two files so that each runs as its own CI shard:
test_scaled_bmm_shapes.py (TestScaledBmmShapes) and test_scaled_bmm_semantics.py
(every other class). Sections 1-10, 13 and 14 of the FP8 scaled_bmm FVT plan;
Sections 11 and 12 are in tests/inductor/test_fp8_operations.py and
tests/inductor/test_fp8_prequantized_load.py. Shapes come from Granite-3.3/4.1-8B-FP8
and Gemma-4-26B-A4B.

Cases that fail today are listed in _KNOWN_FAILURES and run as strict xfails that
also check a fragment of the error message.
Outside test_shapes only the first case of each entry runs (see _cases); the
others come back when the issue is fixed and its entry removed.

TestScaledBmmShapes runs at several core counts, since work division and the
per-core span limit depend on them. The other classes test properties (scales,
out_dtype, boundaries, strides, padding, rejections) that do not depend on the
core count, so they run at CORES_SEMANTIC only.
"""

import contextlib
from typing import NamedTuple

import regex as re

import pytest
import torch
from torch._inductor.exc import InductorError
from utils_inductor import (
    DEVICE,
    _compile_and_run,
    cached_randn,
    strict_xfail,
)

from torch_spyre._inductor import config
from torch_spyre._inductor.constants import FP8_E4M3FN_MAX

FP8 = torch.float8_e4m3fn
FP8_MIN_NORMAL = 2.0**-6
FP8_MIN_SUBNORMAL = 2.0**-9
# Per-element bound: ACC_ULPS x unit roundoff x sum|a*b|. Measured error is at most
# 4.7 unit roundoffs; a mis-laid-out operand gives 57-266. A flat atol/rtol cannot
# separate the two.
ACC_ULPS = 16
# fp16 subnormal spacing, for outputs a small scale pushes toward 0.
FP16_OUTPUT_FLOOR = 4 * 2.0**-24


CORES_BASE = (1, 32)
# Production default; the semantic (non-shape) tests run here only.
CORES_SEMANTIC = (32,)


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


def _params(fvt_ids, *extra, cores=None):
    """pytest params for each case x its core counts (or ``cores`` if given)."""
    tags = [str(e) for e in extra]
    return [
        pytest.param(
            fvt_id, *extra, n, id="-".join([_CASES[fvt_id][0], *tags, f"cores{n}"])
        )
        for fvt_id in fvt_ids
        for n in (cores or _CASES[fvt_id][3])
    ]


class _KnownFailure(NamedTuple):
    """A case that fails today: run as an xfail tied to the open issue that records it.

    ``match`` is a fragment of the error message today. A case that raises ``raises``
    without it fails as StaleKnownFailure, so an entry whose case starts failing for
    another reason fails the run instead of xfailing under the wrong issue.
    ``strict=False`` is only for an outcome that varies between runs.
    """

    test: str
    pattern: str  # case-id regex
    raises: type
    match: str
    reason: str
    strict: bool = True


_SCHEDULER_FAILURE = "sbf-run-scheduler-on-sdsc: failed on program"
_NO_STICK_MECHANISM = "no mechanism to resolve stick incompatibility"
_OUTSIDE_BOUND = "elements outside"

# Order matters: a case takes the first entry that matches it (first match wins),
# both in _expect_known_failure and in _cases' pruning. Put narrower patterns
# before broader ones for the same test.
_KNOWN_FAILURES = [
    _KnownFailure(
        "test_shapes",
        r"3d_qk_32x64x128x64-cores1",
        AssertionError,
        _OUTSIDE_BOUND,
        "Unstable: #6436: [fp8] scaled_bmm: 3D QKᵀ at 1 core returns wrong values "
        "(fails in most runs since #5048, passes in some)",
        strict=False,
    ),
    _KnownFailure(
        "test_shapes",
        r"4d_global_prefill_qk_1x16x24x512x24-cores3",
        AssertionError,
        _OUTSIDE_BOUND,
        "Unstable: #6527: [fp8] scaled_bmm: 4D global-prefill QKᵀ at 3 cores "
        "intermittently returns wrong values (fails in CI, passes locally)",
        strict=False,
    ),
    _KnownFailure(
        "test_shapes",
        r"3d_moe_bmm_as_captured_(8|96)x2816x704x1-cores\d+",
        InductorError,
        _NO_STICK_MECHANISM,
        "#6076: [fp8] scaled_bmm: N=1 MoE expert matmul fails layout planning "
        "when K is not a stick multiple",
    ),
    _KnownFailure(
        "test_shapes",
        r"3d_moe_bmm_as_captured_96x1408x2816x1-cores1"
        r"|3d_moe_bmm_weight_as_mat2_96x1x(2816x1408|704x2816)-cores\d+",
        InductorError,
        "per-core tensor span",
        "#6082: [fp8] scaled_bmm: per-core span limit rejects MoE shapes without "
        "splitting the batch dimension",
    ),
    _KnownFailure(
        "test_shapes",
        r"(3d_moe_bmm_weight_as_mat2_8x1x704x2816"
        r"|4d_decode_pv_(1|4)x32x1x64x128)-cores\d+",
        InductorError,
        _SCHEDULER_FAILURE,
        "#6079: [fp8] scaled_bmm: compilation fails when the contraction K is "
        "not a multiple of 128 (now in the scheduler, after layout planning)",
    ),
    _KnownFailure(
        "test_shapes",
        r"(3d_pv_32x64x64x128|4d_prefill_pv_1x32x64x64x128"
        r"|4d_global_prefill_pv_1x16x(12x12|24x24)x512)-cores\d+",
        InductorError,
        "StopIteration",
        "#6437: [fp8] scaled_bmm: P@V with sub-stick K raises StopIteration in "
        "_cost_model_matmul_planner (no QFP8WT input)",
    ),
    _KnownFailure(
        "test_boundary",
        r".*-tiny-.*",
        AssertionError,
        _OUTSIDE_BOUND,
        "#6067: [fp8] FP8 subnormals mis-decoded in the matmul; #6084: [fp8] "
        "quantize_fp8_with_scale corrupts E4M3 subnormal values",
    ),
    _KnownFailure(
        "test_quantize_subnormals",
        r".*",
        AssertionError,
        "subnormal values flushed to zero by quantize_fp8_with_scale",
        "#6084: [fp8] quantize_fp8_with_scale corrupts E4M3 subnormal values",
    ),
    _KnownFailure(
        "test_padded_inputs",
        r"pad_n_3d_moe_bmm_as_captured_8x2816x704x1-cores\d+",
        InductorError,
        _SCHEDULER_FAILURE,
        "#6079: [fp8] scaled_bmm: compilation fails when the contraction K is "
        "not a multiple of 128 (now in the scheduler, after layout planning)",
    ),
]


class StaleKnownFailure(Exception):
    """A _KNOWN_FAILURES case raised its exception type, but not its message."""


@contextlib.contextmanager
def _expect_known_failure(request):
    """Run the body as an xfail if a _KNOWN_FAILURES entry matches this case.

    The xfail is strict with ``raises=entry.raises``. An error of that type whose
    message lacks ``entry.match`` is re-raised as StaleKnownFailure, which the xfail
    does not accept, so the run fails and the stale entry gets noticed.
    """
    test, case = request.node.originalname, request.node.callspec.id
    entry = next(
        (
            k
            for k in _KNOWN_FAILURES
            if k.test == test and re.fullmatch(k.pattern, case)
        ),
        None,
    )
    if entry is None:
        yield
        return
    if entry.strict:
        strict_xfail(request, raises=entry.raises, reason=entry.reason)
    else:
        request.applymarker(
            pytest.mark.xfail(raises=entry.raises, strict=False, reason=entry.reason)
        )
    try:
        yield
    except entry.raises as e:
        if entry.match not in str(e):
            raise StaleKnownFailure(
                f"{entry.reason}: expected {entry.match!r} in the error, got "
                f"{type(e).__name__}: {str(e)[:500]}"
            ) from e
        raise


# test name -> every case it defines, before _cases prunes repeated known failures.
_ALL_CASES = {}


def _cases(test, params):
    """Register ``test``'s cases and return the ones to run.

    test_shapes runs every case: its known failures depend on the shape and core
    count. Elsewhere a known failure is the same bug for every case it matches and
    a strict xfail already fails the run once the bug is fixed, so only the first
    case of each _KNOWN_FAILURES entry runs. The full lists stay here as data, and
    removing an entry brings all of its cases back as positive tests.
    """
    _ALL_CASES[test] = params
    if test == "test_shapes":
        return params
    seen, kept = set(), []
    for p in params:
        # First match wins, as in _expect_known_failure.
        entry = next(
            (
                i
                for i, k in enumerate(_KNOWN_FAILURES)
                if k.test == test and re.fullmatch(k.pattern, p.id)
            ),
            None,
        )
        if entry is None or entry not in seen:
            kept.append(p)
            if entry is not None:
                seen.add(entry)
    return kept


# Built here, not in test_scaled_bmm_shapes.py, so _ALL_CASES knows the test_shapes
# ids even in the semantics shard, where TestScaledBmmKnownFailures runs.
SHAPE_PARAMS = _cases("test_shapes", _params(_CASES))
