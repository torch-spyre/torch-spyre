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

"""FP8 scaled_bmm FVT for scales, out_dtype, boundaries, rejections, views,
for_each_tile and padding (Sections 6-10, 13 and 14).

These properties do not depend on the core count, so they run at CORES_SEMANTIC
only. Shared cases, helpers and _KNOWN_FAILURES are in scaled_bmm_common.py; the
shape sections are in test_scaled_bmm_shapes.py.
"""

import logging
import os
import sys

import regex as re

import pytest
import torch
from torch._inductor.exc import InductorError
from utils_inductor import (
    DEVICE,
    _compile_and_run,
    cached_randn,
    expects_raise,
)

from torch_spyre._inductor.constants import FP8_E4M3FN_MAX
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.wsr import for_each_tile

sys.path.insert(0, os.path.dirname(__file__))
from scaled_bmm_common import (  # noqa: E402
    CORES_SEMANTIC,
    FP8,
    FP8_MIN_NORMAL,
    FP8_MIN_SUBNORMAL,
    _ALL_CASES,
    _CASES,
    _KNOWN_FAILURES,
    _assert_within_bound,
    _cases,
    _cores,
    _expect_known_failure,
    _fp8_exact,
    _params,
    _run_exact,
    _scaled_bmm_fn,
    _sign,
)


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
        for p in _params(fvt_ids, sa, sb, dt, cores=CORES_SEMANTIC)
    ]


class TestScaledBmmScales:
    """Section 6: scale_a / scale_b are applied inside the op (FR-005, 6.3)."""

    @pytest.mark.parametrize(
        "fvt_id,scale_a,scale_b,scale_dtype,sencores",
        _cases("test_scales", _scale_params()),
    )
    def test_scales(self, fvt_id, scale_a, scale_b, scale_dtype, sencores, request):
        with _expect_known_failure(request):
            _assert_within_bound(
                *_run_exact(fvt_id, sencores, scale_a, scale_b, scale_dtype=scale_dtype)
            )

    @pytest.mark.parametrize(
        "fvt_id,scale_a,scale_b,sencores",
        _cases(
            "test_scale_values",
            [
                p
                for fvt_ids, values in _SCALE_VALUE_GROUPS
                for sa, sb in values
                for p in _params(fvt_ids, sa, sb, cores=CORES_SEMANTIC)
            ],
        ),
    )
    def test_scale_values(self, fvt_id, scale_a, scale_b, sencores):
        """Scales of 0, negative, ~1e-4, 1e-5 and decimals up to 5 (FR-005)."""
        out, ref, absum = _run_scale_value(fvt_id, sencores, scale_a, scale_b)
        assert torch.isfinite(out).all(), "non-finite output"
        _assert_within_bound(out, ref, absum)

    @pytest.mark.parametrize(
        "fvt_id,scale_a_dtype,scale_b_dtype,sencores",
        _cases(
            "test_scale_dtypes",
            [
                p
                for fvt_ids, _ in _SCALE_GROUPS
                for dta, dtb in _SCALE_DTYPE_PAIRS
                for p in _params(fvt_ids, dta, dtb, cores=CORES_SEMANTIC)
            ],
        ),
    )
    def test_scale_dtypes(
        self, fvt_id, scale_a_dtype, scale_b_dtype, sencores, request
    ):
        """Scale dtype pairs with out_dtype left at its default (None), giving fp16."""
        with _expect_known_failure(request):
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
        for p in _params([fvt_id], mode, sa, sb, cores=CORES_SEMANTIC)
    ]


class TestScaledBmmBoundary:
    """Section 7: values near, at, beyond and below the E4M3 range."""

    @pytest.mark.parametrize(
        "fvt_id,mode,scale_a,scale_b,sencores",
        _cases("test_boundary", _boundary_params()),
    )
    def test_boundary(self, fvt_id, mode, scale_a, scale_b, sencores, request):
        with _expect_known_failure(request):
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

    @pytest.mark.parametrize(
        "fvt_id,sencores",
        _cases("test_quantize_subnormals", _params(["SBQK-03"], cores=CORES_SEMANTIC)),
    )
    def test_quantize_subnormals(self, fvt_id, sencores, request):
        """quantize_fp8_with_scale(x, 1.0) keeps FP8-exact values, subnormals included.

        The quantize half of the "tiny" boundary case, without the matmul, so the
        quantizer (#6084) has a signal of its own; test_dma_round_trip_subnormals
        does the same for the matmul's decode (#6067). The result is read back
        through dequantize_fp8_with_scale on the device: a .cpu() copy of the QFP8CH
        quantize output comes back reordered. The first half of the rows hold normal
        values, a control that any other cause of wrong values fails on its own.
        """
        with _expect_known_failure(request):
            _, mat1, _, _ = _CASES[fvt_id]
            u = cached_randn(mat1, differentiation=(fvt_id, "qsub"), abs=True).float()
            u = u.clamp(max=3.0) / 3.0
            normal = torch.zeros(mat1, dtype=torch.bool)
            normal[..., : mat1[-2] // 2, :] = True
            mag = torch.where(
                normal,
                FP8_MIN_NORMAL + (FP8_E4M3FN_MAX - FP8_MIN_NORMAL) * u,
                FP8_MIN_SUBNORMAL + (FP8_MIN_NORMAL - FP8_MIN_SUBNORMAL) * u,
            )
            x = (mag * _sign(mat1, (fvt_id, "qsub"))).to(FP8).float()
            one = torch.tensor(1.0, dtype=torch.float16)

            def fn(t, s):
                q = torch.ops.spyre.quantize_fp8_with_scale(t, s)
                return torch.ops.spyre.dequantize_fp8_with_scale(q, s)

            with _cores(sencores):
                out = _compile_and_run(fn, (x.to(torch.float16), one), DEVICE).float()
            changed = out != x
            n_normal = changed[normal].sum().item()
            assert n_normal == 0, (
                f"{n_normal} / {normal.sum().item()} normal values changed"
            )
            sub = ~normal
            to_other = (changed & sub & (out != 0)).sum().item()
            assert to_other == 0, f"{to_other} subnormal values changed to other values"
            zeroed = (changed & sub).sum().item()
            assert zeroed == 0, (
                f"{zeroed} / {sub.sum().item()} subnormal values flushed to zero by "
                "quantize_fp8_with_scale"
            )


_OUT_DTYPE_GROUPS = [
    ("SBQK-03", (None, torch.float16)),
    ("SB3D-11", (None, torch.float16)),
]

_UNSUPPORTED_OUT_DTYPES = (torch.float32, torch.float8_e4m3fn)

_OUT_DTYPE_NAMES = {
    torch.float32: ("float32", "IEEE_FP32"),
    torch.float8_e4m3fn: ("float8_e4m3fn", "SEN143_FP8"),
}


class TestScaledBmmParameters:
    """Section 8: out_dtype resolution and the ignored use_fast_accum flag."""

    @pytest.mark.parametrize(
        "fvt_id,out_dtype,sencores",
        _cases(
            "test_out_dtype",
            [
                p
                for fvt_id, dtypes in _OUT_DTYPE_GROUPS
                for dt in dtypes
                for p in _params([fvt_id], dt, cores=CORES_SEMANTIC)
            ],
        ),
    )
    def test_out_dtype(self, fvt_id, out_dtype, sencores, request):
        """out_dtype=None and out_dtype=torch.float16 both give fp16 (FRS 5.1)."""
        with _expect_known_failure(request):
            expected = out_dtype or torch.float16
            out, ref, absum = _run_exact(fvt_id, sencores, out_dtype=out_dtype)
            assert out.dtype == expected, f"out_dtype={out_dtype} gave {out.dtype}"
            _assert_within_bound(out, ref, absum, expected)

    @pytest.mark.parametrize(
        "fvt_id,sencores",
        _cases("test_out_dtype_bfloat16", _params(["SBQK-03"], cores=CORES_SEMANTIC)),
    )
    def test_out_dtype_bfloat16(self, fvt_id, sencores, request):
        """out_dtype=bfloat16 compiles; the output must be bf16 or fp16 (SBREJ-12).

        Fails today on #6061: scaled_bmm_decomp multiplies the bf16 bmm_fp8 result
        by the 0-d fp16 scales, and that multiply is promoted to float32.
        """
        with _expect_known_failure(request):
            out, ref, absum = _run_exact(fvt_id, sencores, out_dtype=torch.bfloat16)
            assert out.dtype in (torch.float16, torch.bfloat16), (
                f"out_dtype=torch.bfloat16 returned {out.dtype}"
            )
            _assert_within_bound(out, ref, absum, out.dtype)

    @pytest.mark.parametrize(
        "fvt_id,use_fast_accum,sencores",
        _cases(
            "test_use_fast_accum_warns",
            [
                p
                for fvt_id in ("SBQK-03", "SB3D-11")
                for ufa in (True, False)
                for p in _params([fvt_id], ufa, cores=CORES_SEMANTIC)
            ],
        ),
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
    ("rank2", (128, 128), (128, 128), FP8, FP8, "bmm_fp8 with shapes"),
    (
        "rank5",
        (1, 1, 32, 128, 128),
        (1, 1, 32, 128, 128),
        FP8,
        FP8,
        "bmm_fp8 with shapes",
    ),
    ("rank_mismatch_4_3", _G4D, (32, 128, 128), FP8, FP8, "bmm_fp8 with shapes"),
    ("batch_mismatch", (2, 32, 128, 128), _G4D, FP8, FP8, "bmm_fp8 with shapes"),
    ("gqa_heads_unexpanded", _G4D, (1, 8, 128, 128), FP8, FP8, "bmm_fp8 with shapes"),
    (
        "broadcast_batch",
        (2, 32, 128, 128),
        (1, 1, 128, 128),
        FP8,
        FP8,
        "bmm_fp8 with shapes",
    ),
    ("k_mismatch", _G4D, (1, 32, 256, 128), FP8, FP8, "bmm_fp8 with shapes"),
    (
        "gqa_unexpanded_global",
        (1, 16, 64, 512),
        (1, 2, 512, 2048),
        FP8,
        FP8,
        "bmm_fp8 with shapes",
    ),
    (
        "gqa_unexpanded_sliding",
        (1, 16, 64, 256),
        (1, 8, 256, 1024),
        FP8,
        FP8,
        "bmm_fp8 with shapes",
    ),
]


class TestScaledBmmRejections:
    """Section 9: invalid inputs raise instead of returning a tensor (9.1)."""

    @expects_raise
    @pytest.mark.parametrize(
        "key,mat1,mat2,dt1,dt2,fragment,sencores",
        _cases(
            "test_rejections",
            [
                pytest.param(*c, n, id=f"{c[0]}-cores{n}")
                for c in REJECTION_CASES
                for n in CORES_SEMANTIC
            ],
        ),
    )
    def test_rejections(self, key, mat1, mat2, dt1, dt2, fragment, sencores, request):
        with _expect_known_failure(request):
            a = cached_randn(mat1, differentiation=(key, "mat1")).to(dt1)
            b = cached_randn(mat2, differentiation=(key, "mat2")).to(dt2)

            def fn(a, b):
                return torch.ops.spyre.scaled_bmm(a, b, out_dtype=torch.float16)

            with _cores(sencores), pytest.raises(InductorError, match=fragment):
                _compile_and_run(fn, (a, b), DEVICE)

    @expects_raise
    @pytest.mark.parametrize(
        "fvt_id,out_dtype,sencores",
        _cases(
            "test_rejects_out_dtype",
            [
                p
                for dt in _UNSUPPORTED_OUT_DTYPES
                for p in _params(["SBQK-03"], dt, cores=CORES_SEMANTIC)
            ],
        ),
    )
    def test_rejects_out_dtype(self, fvt_id, out_dtype, sencores):
        """float32 and FP8 out_dtype raise an error naming the dtype (SBREJ-13, 14).

        bfloat16 is not rejected; see test_out_dtype_bfloat16 (SBREJ-12).
        """
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


# (heads, head_dim)
_VIEW_GEOMETRIES = [(32, 128), (16, 256)]
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
        "heads,head_dim,variant,sencores",
        _cases(
            "test_non_contiguous_views",
            [
                pytest.param(h, d, v, n, id=f"{h}x{d}-{v}-cores{n}")
                for h, d in _VIEW_GEOMETRIES
                for v in _VIEW_VARIANTS
                for n in CORES_SEMANTIC
            ],
        ),
    )
    def test_non_contiguous_views(self, heads, head_dim, variant, sencores):
        """The view operand matches the CPU result and its contiguous copy."""
        ra, rb = _view_inputs(variant, heads, head_dim)
        a, b = ra.to(torch.float16), rb.to(torch.float16)  # keeps the strides
        assert not (a.is_contiguous() and b.is_contiguous())
        one = torch.tensor(1.0, dtype=torch.float16)
        fn = _scaled_bmm_fn()
        with _cores(sencores):
            out = _compile_and_run(fn, (a, b, one, one, one, one), DEVICE)
            ctrl = _compile_and_run(
                fn, (a.contiguous(), b.contiguous(), one, one, one, one), DEVICE
            )
        _assert_within_bound(out, ra @ rb, ra.abs() @ rb.abs())
        assert torch.equal(out, ctrl), "view result differs from contiguous"


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


# (fvt id, test key, B, M, D, page N)
_FET_PAGED = [
    ("SBFT-01", "paged_qk_granite_decode_32x1x128", 32, 1, 128, 128),
    ("SBFT-02", "paged_qk_granite_prefill_32x512x128", 32, 512, 128, 128),
    ("SBFT-03", "paged_pv_granite_32x1x128", 32, 1, 128, 128),
    ("SBFT-04", "paged_qk_gemma_global_16x1x512", 16, 1, 512, 128),
    ("SBFT-05", "paged_qk_gemma_sliding_16x1x256", 16, 1, 256, 128),
]


def _fet_params():
    out = [
        pytest.param(fvt_id, key, b, m, d, n, c, id=f"{key}-cores{c}")
        for fvt_id, key, b, m, d, n in _FET_PAGED
        for c in CORES_SEMANTIC
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
        for c in CORES_SEMANTIC
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

    @pytest.mark.parametrize(
        "fvt_id,key,b,m,d,n,sencores", _cases("test_for_each_tile", _fet_params())
    )
    def test_for_each_tile(self, fvt_id, key, b, m, d, n, sencores, request):
        """The FP8 block inside for_each_tile gives the CPU result.

        Today every case fails on #5423 (batchmatmulfp8 layout planning inside
        for_each_tile), so this test exercises only that bug, and one case runs.
        """
        with _expect_known_failure(request):
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
# (FVT id, test key, source case, axis, padded extent)
_PAD_CASES = [
    ("SBPAD-01", "pad_k_4d_prefill_pv_1x32x64x64x128", "SBPV-01", "k", 128),
    (
        "SBPAD-02",
        "pad_k_4d_global_prefill_pv_1x16x12x12x512",
        "SBMDL-06",
        "k",
        128,
    ),
    (
        "SBPAD-03",
        "pad_k_3d_moe_bmm_weight_as_mat2_8x1x704x2816",
        "SB3D-09",
        "k",
        768,
    ),
    (
        "SBPAD-04",
        "pad_n_3d_moe_bmm_as_captured_8x2816x704x1",
        "SB3D-08",
        "n",
        64,
    ),
    (
        "SBPAD-05",
        "pad_n_4d_global_prefill_qk_1x16x12x512x12",
        "SBMDL-05",
        "n",
        64,
    ),
    (
        "SBPAD-06",
        "pad_m_4d_granite-3.3-8b-instruct-FP8_prefill_qk_1x32x41x128x2048",
        "SBMDL-01",
        "m",
        64,
    ),
]


class TestScaledBmmPadded:
    """Section 13: zero-padded operands give the unpadded result, pad region zero."""

    @pytest.mark.parametrize(
        "fvt_id,key,source,axis,extent,sencores",
        _cases(
            "test_padded_inputs",
            [
                pytest.param(f, k, s, ax, e, c, id=f"{k}-cores{c}")
                for f, k, s, ax, e in _PAD_CASES
                for c in CORES_SEMANTIC
            ],
        ),
    )
    def test_padded_inputs(self, fvt_id, key, source, axis, extent, sencores, request):
        """The live part matches the unpadded CPU result; the padding is zero."""
        with _expect_known_failure(request):
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


class TestScaledBmmKnownFailures:
    """Every _KNOWN_FAILURES entry matches a case and says what it expects."""

    @pytest.mark.parametrize(
        "entry",
        [pytest.param(k, id=f"{k.test}-{i}") for i, k in enumerate(_KNOWN_FAILURES)],
    )
    def test_entry_matches_a_case(self, entry):
        """A stale regex is silently a no-op; an empty fragment accepts any error."""
        assert entry.test in _ALL_CASES, f"no parametrized test named {entry.test}"
        ids = [p.id for p in _ALL_CASES[entry.test]]
        assert any(re.fullmatch(entry.pattern, i) for i in ids), (
            f"{entry.test}: {entry.pattern!r} matches none of its {len(ids)} cases"
        )
        assert entry.match, f"{entry.test}: {entry.pattern!r} has no message fragment"
        assert re.search(r"#\d+", entry.reason), f"{entry.reason!r} cites no issue"
        assert entry.strict == (not entry.reason.startswith("Unstable: ")), (
            "a non-strict entry's reason starts with 'Unstable: ', and only its"
        )
