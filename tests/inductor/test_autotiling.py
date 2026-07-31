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

import gc
import os
import pytest
import torch
import torch._inductor.codecache
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
import utils_inductor
from torch_spyre._inductor import config as spyre_config
import torch_spyre._inductor.wsr.propagate_named_dims as _pnd

_TOLERANCES = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1e-2, 1e-2),
    torch.float32: (1e-5, 1e-5),
}

# ---------------------------------------------------------------------------
# Shared xfail reasons — keyed to GitHub issues.
# ---------------------------------------------------------------------------
# Bug 1: dxp_standalone SIGABRT in LoopCoalescing on Fermat-prime row counts
#        https://github.com/torch-spyre/torch-spyre/issues/3413
_FERMAT_PRIME_XFAIL = (
    "xfail",
    "dxp_standalone SIGABRT: LoopCoalescing cannot factor Fermat-prime tile count (issue #3413)",
)

# Bug 2: dxp_standalone SIGABRT: L3_ADDEARIMM immediate value out of boundary
#        https://github.com/torch-spyre/torch-spyre/issues/3414
_L3_ADDEARIMM_XFAIL = (
    "xfail",
    "dxp_standalone SIGABRT: L3_ADDEARIMM immediate value out of boundary (issue #3414)",
)

# Bug 3: work_division silently proceeds past span limit → wrong result or NaN
#        https://github.com/torch-spyre/torch-spyre/issues/3415
_WORK_DIV_XFAIL = (
    "xfail",
    "work_division proceeds past span limit; wrong result or NaN (issue #3415)",
)

# Bug 4: span-overflow planner exhausts 512 split candidates → Unsupported
#        https://github.com/torch-spyre/torch-spyre/issues/3417
_PLANNER_XFAIL = (
    "xfail",
    "span-overflow planner exhausts 512 split candidates; Unsupported raised (issue #3417)",
)


def _tol(dtype):
    """Return (atol, rtol) for the given dtype."""
    return _TOLERANCES[dtype]


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
def _residual_add(x, y):
    """x + y -- binary elementwise."""
    return x + y


def _rmsnorm_gamma_scale(weight, x):
    """weight * x -- binary elementwise."""
    return weight * x


def _logit_temperature(x):
    """x / 16.0 -- scalar elementwise."""
    return x / 16.0


def _embedding_scale(x):
    """x * 12.0 -- scalar elementwise."""
    return x * 12.0


def _sdpa_scale_standalone(x):
    """x * 0.0078125 -- scalar elementwise."""
    return x * 0.0078125


def _rope_cos_mul(x, cos):
    """x * cos -- binary elementwise."""
    return x * cos


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _rope_sin_mul(x, sin):
    """rotate_half(x) * sin -- binary elementwise."""
    return _rotate_half(x) * sin


def _swiglu_silu(gate):
    """F.silu(gate) -- unary elementwise."""
    return F.silu(gate)


def _swiglu_gate_mul(gate, up):
    """F.silu(gate) * up -- binary elementwise."""
    return F.silu(gate) * up


def _yarn_floor(divided_pos):
    """floor(x) -- unary elementwise."""
    return torch.floor(divided_pos)


def _yarn_log_scale(divided_pos):
    return torch.log(torch.floor(torch.abs(divided_pos)) + 1)


def _vision_gelu(x):
    """F.gelu(x) -- unary elementwise."""
    return F.gelu(x)


def _vision_silu(x):
    """F.silu(x) -- unary elementwise."""
    return F.silu(x)


def _torch_add(x, y):
    """torch.add(x, y) -- binary elementwise. Same-shape assumption."""
    return torch.add(x, y)


def _torch_mul(x, y):
    """torch.mul(x, y) -- binary elementwise. Same-shape assumption."""
    return torch.mul(x, y)


def _torch_pow(x):
    """torch.pow(x, 2.0) -- unary elementwise. Exponent assumed to be a
    fixed scalar (2.0/squaring), since the doc gives no exponent value."""
    return torch.pow(x, 2.0)


def _torch_rsqrt(x):
    """torch.rsqrt(x) -- unary elementwise. rsqrt is undefined for x <= 0,
    so abs()+eps keeps randomly generated input tensors in-domain."""
    return torch.rsqrt(torch.abs(x) + 1e-6)


def _torch_neg(x):
    """torch.neg(x) -- unary elementwise."""
    return torch.neg(x)


def _mk(shape, dtype):
    return torch.randn(*shape, dtype=dtype)


_OPS = {
    "residual_add": (_residual_add, "binary_same"),
    "rmsnorm_gamma_scale": (_rmsnorm_gamma_scale, "binary_weight_lastdim"),
    "logit_temperature": (_logit_temperature, "unary"),
    "embedding_scale": (_embedding_scale, "unary"),
    "sdpa_scale_standalone": (_sdpa_scale_standalone, "unary"),
    "rope_cos_mul": (_rope_cos_mul, "binary_rope"),
    "rope_sin_mul": (_rope_sin_mul, "binary_rope"),
    "swiglu_silu": (_swiglu_silu, "unary"),
    "swiglu_gate_mul": (_swiglu_gate_mul, "binary_same"),
    "yarn_floor": (_yarn_floor, "unary"),
    "yarn_log_scale": (_yarn_log_scale, "unary"),
    "vision_gelu": (_vision_gelu, "unary"),
    "vision_silu": (_vision_silu, "unary"),
    "torch_add": (_torch_add, "binary_same"),
    "torch_mul": (_torch_mul, "binary_same"),
    "torch_pow": (_torch_pow, "unary"),
    "torch_rsqrt": (_torch_rsqrt, "unary"),
    "torch_neg": (_torch_neg, "unary"),
}


def _build_inputs(op_key, shape, dtype):
    """Build (fn, a, b) for op_key at the given shape/dtype."""
    fn, kind = _OPS[op_key]
    x = _mk(shape, dtype)
    if kind == "unary":
        return fn, x, None
    if kind == "binary_same":
        y = _mk(shape, dtype)
        return fn, x, y
    if kind == "binary_weight_lastdim":
        weight = _mk((shape[-1],), dtype)
        return fn, weight, x
    if kind == "binary_rope":
        aux_shape = (1, 1) + tuple(shape[2:])
        aux = _mk(aux_shape, dtype)
        return fn, x, aux
    raise ValueError(f"Unknown op kind for {op_key!r}")


_DTYPE_SHORT = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


def _shape_str(shape):
    return "x".join(str(d) for d in shape)


def _expand_rows(rows, expected_failures=None):
    ef = expected_failures or {}
    out = []
    for entry in rows:
        op_key, shape, dtypes, cores_list = entry[:4]
        prefix = entry[4] if len(entry) > 4 else f"{op_key}-{_shape_str(shape)}"
        dtype_list = dtypes if isinstance(dtypes, (list, tuple)) else [dtypes]
        for dtype in dtype_list:
            for c in cores_list:
                pid = f"{prefix}-{_DTYPE_SHORT[dtype]}-cores{c}"
                marks = []
                if pid in ef:
                    act, reason = ef[pid]
                    marks.append(
                        pytest.mark.skip(reason=reason)
                        if act == "skip"
                        else pytest.mark.xfail(reason=reason)
                    )
                out.append(pytest.param(op_key, shape, dtype, c, id=pid, marks=marks))
    return out


# =============================================================================
# S0 -- Baseline op-formula correctness (never overflows).
# =============================================================================
S0_ROWS = [
    ("yarn_floor", (16,), torch.float32, [2, 9, 15, 22, 30]),
    ("vision_gelu", (512, 5120), torch.bfloat16, [3, 8, 18, 26, 32]),
    ("residual_add", (1, 12, 4096), torch.bfloat16, [1, 5, 11, 19, 28]),
    (
        "embedding_scale",
        (1, 64, 4096),
        [torch.bfloat16, torch.float16],
        [5, 11, 20, 27, 32],
    ),
    ("logit_temperature", (2, 41, 49152), torch.bfloat16, [1, 8, 16, 25, 31]),
    # fp32 excluded: silu on DataFormats.IEEE_FP32 is unsupported on Spyre hardware.
    ("swiglu_silu", (2, 16, 14336), torch.bfloat16, [4, 10, 18, 26, 32]),
    ("swiglu_gate_mul", (1, 8, 14336), torch.bfloat16, [1, 6, 15, 23, 29]),
    (
        "vision_silu",
        (1, 1024, 4096),
        [torch.bfloat16, torch.float16],
        [4, 12, 19, 27, 31],
    ),
    (
        "rope_cos_mul",
        (1, 32, 16, 128),
        [torch.bfloat16, torch.float16],
        [2, 7, 14, 22, 30],
    ),
    ("rope_sin_mul", (2, 32, 12, 128), torch.bfloat16, [3, 9, 17, 24, 31]),
    (
        "sdpa_scale_standalone",
        (1, 40, 64, 128),
        [torch.bfloat16, torch.float32],
        [6, 13, 21, 28, 32],
    ),
]
EXPECTED_FAILURES_S0 = {}
# =============================================================================
# S1 -- Pass-through / negative result.
# =============================================================================
S1_ROWS = [
    ("yarn_log_scale", (16,), torch.float16, [3, 10, 17, 24, 31]),
    (
        "residual_add",
        (1, 1, 4096),
        torch.bfloat16,
        [1, 7, 14, 21, 32],
        "residual_add_decode",
    ),
]
EXPECTED_FAILURES_S1 = {}
# =============================================================================
# S4 -- Boundary
# =============================================================================
S4_ROWS = [
    ("residual_add", (32768, 4096), torch.bfloat16, [1], "exact_2d_a"),
    ("residual_add", (32769, 4096), torch.bfloat16, [1], "just_above_2d_a"),
    ("logit_temperature", (65536, 2048), torch.bfloat16, [1], "exact_2d_b"),
    ("logit_temperature", (65537, 2048), torch.bfloat16, [1], "just_above_2d_b"),
    ("rmsnorm_gamma_scale", (512, 512, 512), torch.bfloat16, [1], "exact_3d_a"),
    ("rmsnorm_gamma_scale", (513, 512, 512), torch.bfloat16, [1], "just_above_3d_a"),
    ("residual_add", (1024, 256, 512), torch.bfloat16, [1], "exact_3d_b"),
    ("residual_add", (1025, 256, 512), torch.bfloat16, [1], "just_above_3d_b"),
    ("embedding_scale", (128, 32, 64, 512), torch.bfloat16, [1], "exact_4d_a"),
    ("embedding_scale", (129, 32, 64, 512), torch.bfloat16, [1], "just_above_4d_a"),
    ("rope_cos_mul", (256, 16, 128, 256), torch.bfloat16, [1], "exact_4d_b"),
    ("rope_cos_mul", (257, 16, 128, 256), torch.bfloat16, [1], "just_above_4d_b"),
]
EXPECTED_FAILURES_S4 = {
    # Bug 1 — issue #3413
    "just_above_2d_b-bf16-cores1": _FERMAT_PRIME_XFAIL,
    # Bug 1 — issue #3413
    "just_above_4d_b-bf16-cores1": _FERMAT_PRIME_XFAIL,
}
# =============================================================================
# S2 - per-core span overflow.
# =============================================================================
S2_ROWS = [
    ("residual_add", (130000, 4096), torch.bfloat16, [1, 2, 3]),
    ("residual_add", (130000, 4096), torch.float16, [1, 2, 3]),
    ("residual_add", (130000, 4096), torch.float32, [1, 2, 3, 4, 5, 6, 7]),
    ("rmsnorm_gamma_scale", (120000, 4096), torch.bfloat16, [1, 2, 3]),
    ("rmsnorm_gamma_scale", (120000, 4096), torch.float16, [1, 2, 3]),
    ("rmsnorm_gamma_scale", (120000, 4096), torch.float32, [1, 2, 3, 4, 5, 6]),
    ("residual_add", (135000, 4096), torch.bfloat16, [1, 2, 3]),
    ("residual_add", (135000, 4096), torch.float16, [1, 2, 3]),
    ("residual_add", (195000, 2880), torch.bfloat16, [1, 2, 3]),
    ("residual_add", (195000, 2880), torch.float16, [1, 2, 3]),
    ("residual_add", (195000, 2880), torch.float32, [1, 2, 3, 4, 5, 6, 7]),
    ("residual_add", (185000, 2880), torch.bfloat16, [1, 2, 3]),
    ("residual_add", (185000, 2880), torch.float16, [1, 2, 3]),
    ("residual_add", (48000, 2880), torch.float32, [1]),
    ("embedding_scale", (50000, 2880), torch.bfloat16, [1]),
    ("embedding_scale", (50000, 2880), torch.float16, [1]),
    ("embedding_scale", (50000, 2880), torch.float32, [1, 2]),
    ("residual_add", (125000, 5120), torch.bfloat16, [1, 2, 3, 4]),
    ("residual_add", (125000, 5120), torch.float16, [1, 2, 3, 4]),
    ("residual_add", (125000, 5120), torch.float32, [1, 2, 3, 5, 6, 8, 9]),
    ("rmsnorm_gamma_scale", (140000, 5120), torch.bfloat16, [1, 2, 3, 4, 5]),
    ("rmsnorm_gamma_scale", (140000, 5120), torch.float16, [1, 2, 3, 4, 5]),
    ("rmsnorm_gamma_scale", (140000, 5120), torch.float32, [1, 2, 4, 5, 7, 8, 10]),
    ("residual_add", (115000, 5120), torch.bfloat16, [1, 2, 3, 4]),
    ("residual_add", (115000, 5120), torch.float16, [1, 2, 3, 4]),
    ("residual_add", (98000, 4096), torch.bfloat16, [1, 2]),
    ("residual_add", (98000, 4096), torch.float16, [1, 2]),
    ("residual_add", (98000, 4096), torch.float32, [1, 2, 3, 4, 5]),
    ("logit_temperature", (105000, 4096), torch.bfloat16, [1, 2, 3]),
    ("logit_temperature", (105000, 4096), torch.float16, [1, 2, 3]),
    ("logit_temperature", (105000, 4096), torch.float32, [1, 2, 3, 4, 5, 6]),
    ("logit_temperature", (92000, 4096), torch.bfloat16, [1, 2]),
    ("logit_temperature", (92000, 4096), torch.float16, [1, 2]),
    ("residual_add", (47000, 4096), torch.bfloat16, [1]),
    ("residual_add", (47000, 4096), torch.float16, [1]),
    ("residual_add", (47000, 4096), torch.float32, [1, 2]),
    ("logit_temperature", (52000, 4096), torch.bfloat16, [1]),
    ("logit_temperature", (52000, 4096), torch.float16, [1]),
    ("logit_temperature", (52000, 4096), torch.float32, [1, 2, 3]),
    ("logit_temperature", (45000, 4096), torch.bfloat16, [1]),
    ("logit_temperature", (45000, 4096), torch.float16, [1]),
    ("residual_add", (2750, 12, 4096), torch.float32, [1]),
    ("residual_add", (2800, 16, 4096), torch.bfloat16, [1]),
    ("residual_add", (2800, 16, 4096), torch.float16, [1]),
    ("residual_add", (2800, 16, 4096), torch.float32, [1, 2]),
    ("logit_temperature", (75, 41, 49152), torch.bfloat16, [1]),
    ("logit_temperature", (75, 41, 49152), torch.float16, [1]),
    ("logit_temperature", (75, 41, 49152), torch.float32, [1, 2]),
    ("logit_temperature", (65, 44, 49152), torch.float32, [1]),
    ("rope_cos_mul", (6144, 32, 12, 128), torch.bfloat16, [1, 2]),
    ("rope_cos_mul", (6144, 32, 12, 128), torch.float16, [1, 2]),
    ("rope_cos_mul", (6144, 32, 12, 128), torch.float32, [1, 2, 3, 4]),
    ("rope_cos_mul", (10000, 32, 12, 128), torch.bfloat16, [1, 2, 3]),
    ("rope_cos_mul", (10000, 32, 12, 128), torch.float16, [1, 2, 3]),
    ("rope_cos_mul", (10000, 32, 12, 128), torch.float32, [1, 2, 3, 4, 5, 6]),
    ("rope_cos_mul", (28000, 32, 12, 128), torch.bfloat16, [1, 2, 3, 5, 6, 8, 9]),
    ("rope_cos_mul", (28000, 32, 12, 128), torch.float16, [1, 2, 3, 5, 6, 8, 9]),
    ("rope_cos_mul", (28000, 32, 12, 128), torch.float32, [1, 2, 5, 9, 12, 16, 19]),
    ("rope_sin_mul", (30000, 32, 16, 128), torch.bfloat16, [1, 2, 4, 6, 9, 11, 13]),
    ("rope_sin_mul", (30000, 32, 16, 128), torch.float16, [1, 2, 4, 6, 9, 11, 13]),
    ("rope_sin_mul", (30000, 32, 16, 128), torch.float32, [1, 2, 7, 12, 17, 22, 27]),
    ("torch_add", (125, 1064, 1024), torch.float32, [1]),
    ("torch_add", (500, 1064, 1024), torch.bfloat16, [1, 2, 3]),
    ("torch_add", (500, 1064, 1024), torch.float16, [1, 2, 3]),
    ("torch_add", (500, 1064, 1024), torch.float32, [1, 2, 3, 4, 5, 6, 7]),
    ("torch_add", (1000, 1064, 1024), torch.bfloat16, [1, 2, 3, 4, 5, 6, 7]),
    ("torch_add", (1000, 1064, 1024), torch.float16, [1, 2, 3, 4, 5, 6, 7]),
    ("torch_add", (1000, 1064, 1024), torch.float32, [1, 2, 5, 7, 10, 12, 15]),
    ("torch_mul", (250, 1064, 1024), torch.bfloat16, [1]),
    ("torch_mul", (250, 1064, 1024), torch.float16, [1]),
    ("torch_mul", (250, 1064, 1024), torch.float32, [1, 2, 3]),
    ("torch_mul", (750, 1064, 1024), torch.bfloat16, [1, 2, 3, 4, 5]),
    ("torch_mul", (750, 1064, 1024), torch.float16, [1, 2, 3, 4, 5]),
    ("torch_mul", (750, 1064, 1024), torch.float32, [1, 2, 4, 6, 7, 9, 11]),
    ("torch_pow", (300, 1064, 1024), torch.bfloat16, [1, 2]),
    ("torch_pow", (300, 1064, 1024), torch.float16, [1, 2]),
    ("torch_pow", (300, 1064, 1024), torch.float32, [1, 2, 3, 4]),
    ("torch_pow", (800, 1064, 1024), torch.bfloat16, [1, 2, 3, 4, 5, 6]),
    ("torch_pow", (800, 1064, 1024), torch.float16, [1, 2, 3, 4, 5, 6]),
    ("torch_pow", (800, 1064, 1024), torch.float32, [1, 2, 4, 6, 8, 10, 12]),
    ("torch_rsqrt", (400, 1064, 1024), torch.bfloat16, [1, 2, 3]),
    ("torch_rsqrt", (400, 1064, 1024), torch.float16, [1, 2, 3]),
    ("torch_rsqrt", (400, 1064, 1024), torch.float32, [1, 2, 3, 4, 5, 6]),
    ("torch_rsqrt", (600, 1064, 1024), torch.bfloat16, [1, 2, 3, 4]),
    ("torch_rsqrt", (600, 1064, 1024), torch.float16, [1, 2, 3, 4]),
    ("torch_rsqrt", (600, 1064, 1024), torch.float32, [1, 2, 3, 5, 6, 8, 9]),
    ("torch_add", (125, 16, 1064, 64), torch.float32, [1]),
    ("torch_add", (500, 16, 1064, 64), torch.bfloat16, [1, 2, 3]),
    ("torch_add", (500, 16, 1064, 64), torch.float16, [1, 2, 3]),
    ("torch_add", (500, 16, 1064, 64), torch.float32, [1, 2, 3, 4, 5, 6, 7]),
    ("torch_mul", (250, 16, 1064, 64), torch.bfloat16, [1]),
    ("torch_mul", (250, 16, 1064, 64), torch.float16, [1]),
    ("torch_mul", (250, 16, 1064, 64), torch.float32, [1, 2, 3]),
    ("torch_mul", (1000, 16, 1064, 64), torch.bfloat16, [1, 2, 3, 4, 5, 6, 7]),
    ("torch_mul", (1000, 16, 1064, 64), torch.float16, [1, 2, 3, 4, 5, 6, 7]),
    ("torch_mul", (1000, 16, 1064, 64), torch.float32, [1, 2, 5, 7, 10, 12, 15]),
    ("residual_add", (32, 64, 1024, 2049), torch.float16, [1, 2, 8, 13, 19, 24, 30]),
    # Meta-Llama-3.1-8B  (128256, 4096)  bf16/fp16 overflow=[1,2,3]  fp32=[1..7]
    (
        "residual_add",
        (128256, 4096),
        torch.bfloat16,
        [1, 2, 3],
        "llama3_exact-residual_add",
    ),
    (
        "residual_add",
        (128256, 4096),
        torch.float16,
        [1, 2, 3],
        "llama3_exact-residual_add",
    ),
    (
        "residual_add",
        (128256, 4096),
        torch.float32,
        [1, 2, 4, 5, 7],
        "llama3_exact-residual_add",
    ),
    (
        "logit_temperature",
        (128256, 4096),
        torch.bfloat16,
        [1, 2, 3],
        "llama3_exact-logit_temperature",
    ),
    (
        "logit_temperature",
        (128256, 4096),
        torch.float16,
        [1, 2, 3],
        "llama3_exact-logit_temperature",
    ),
    (
        "logit_temperature",
        (128256, 4096),
        torch.float32,
        [1, 2, 4, 5, 7],
        "llama3_exact-logit_temperature",
    ),
    # gpt-oss-20b  (201088, 2880)  bf16/fp16 overflow=[1..4]  fp32=[1..8]
    (
        "residual_add",
        (201088, 2880),
        torch.bfloat16,
        [1, 2, 3, 4],
        "gptoss20b_exact-residual_add",
    ),
    (
        "residual_add",
        (201088, 2880),
        torch.float16,
        [1, 2, 3, 4],
        "gptoss20b_exact-residual_add",
    ),
    (
        "residual_add",
        (201088, 2880),
        torch.float32,
        [1, 2, 4, 5, 7, 8],
        "gptoss20b_exact-residual_add",
    ),
    (
        "embedding_scale",
        (201088, 2880),
        torch.bfloat16,
        [1, 2, 3, 4],
        "gptoss20b_exact-embedding_scale",
    ),
    (
        "embedding_scale",
        (201088, 2880),
        torch.float16,
        [1, 2, 3, 4],
        "gptoss20b_exact-embedding_scale",
    ),
    (
        "embedding_scale",
        (201088, 2880),
        torch.float32,
        [1, 2, 4, 5, 7, 8],
        "gptoss20b_exact-embedding_scale",
    ),
    # Ministral-3B / Mistral-24B  (131072, 5120)  bf16/fp16 overflow=[1..4]  fp32=[1..9]
    (
        "residual_add",
        (131072, 5120),
        torch.bfloat16,
        [1, 2, 3, 4],
        "mistral_exact-residual_add",
    ),
    (
        "residual_add",
        (131072, 5120),
        torch.float16,
        [1, 2, 3, 4],
        "mistral_exact-residual_add",
    ),
    (
        "residual_add",
        (131072, 5120),
        torch.float32,
        [1, 2, 4, 5, 8, 9],
        "mistral_exact-residual_add",
    ),
    (
        "rmsnorm_gamma_scale",
        (131072, 5120),
        torch.bfloat16,
        [1, 2, 3, 4],
        "mistral_exact-rmsnorm_gamma_scale",
    ),
    (
        "rmsnorm_gamma_scale",
        (131072, 5120),
        torch.float16,
        [1, 2, 3, 4],
        "mistral_exact-rmsnorm_gamma_scale",
    ),
    (
        "rmsnorm_gamma_scale",
        (131072, 5120),
        torch.float32,
        [1, 2, 4, 5, 8, 9],
        "mistral_exact-rmsnorm_gamma_scale",
    ),
    # granite-4.1-8b  (100352, 4096)  bf16/fp16 overflow=[1,2,3]  fp32=[1..6]
    (
        "residual_add",
        (100352, 4096),
        torch.bfloat16,
        [1, 2, 3],
        "granite41_exact-residual_add",
    ),
    (
        "residual_add",
        (100352, 4096),
        torch.float16,
        [1, 2, 3],
        "granite41_exact-residual_add",
    ),
    (
        "residual_add",
        (100352, 4096),
        torch.float32,
        [1, 2, 4, 5, 6],
        "granite41_exact-residual_add",
    ),
    (
        "logit_temperature",
        (100352, 4096),
        torch.bfloat16,
        [1, 2, 3],
        "granite41_exact-logit_temperature",
    ),
    (
        "logit_temperature",
        (100352, 4096),
        torch.float16,
        [1, 2, 3],
        "granite41_exact-logit_temperature",
    ),
    (
        "logit_temperature",
        (100352, 4096),
        torch.float32,
        [1, 2, 4, 5, 6],
        "granite41_exact-logit_temperature",
    ),
    # llama3_big  (256512, 4096) = 2×128256  bf16/fp16=[1..7]  fp32=[1..15]
    (
        "residual_add",
        (256512, 4096),
        torch.bfloat16,
        [1, 2, 4, 5, 6, 7],
        "llama3_big-residual_add",
    ),
    (
        "residual_add",
        (256512, 4096),
        torch.float16,
        [1, 2, 4, 5, 6, 7],
        "llama3_big-residual_add",
    ),
    (
        "residual_add",
        (256512, 4096),
        torch.float32,
        [1, 2, 3, 6, 9, 14, 15],
        "llama3_big-residual_add",
    ),
    (
        "logit_temperature",
        (256512, 4096),
        torch.bfloat16,
        [1, 2, 4, 5, 6, 7],
        "llama3_big-logit_temperature",
    ),
    (
        "logit_temperature",
        (256512, 4096),
        torch.float16,
        [1, 2, 4, 5, 6, 7],
        "llama3_big-logit_temperature",
    ),
    (
        "logit_temperature",
        (256512, 4096),
        torch.float32,
        [1, 2, 3, 6, 9, 14, 15],
        "llama3_big-logit_temperature",
    ),
    # gptoss20b_big  (402176, 2880) = 2×201088  bf16/fp16=[1..8]  fp32=[1..17]
    (
        "residual_add",
        (402176, 2880),
        torch.bfloat16,
        [1, 2, 3, 5, 7, 8],
        "gptoss20b_big-residual_add",
    ),
    (
        "residual_add",
        (402176, 2880),
        torch.float16,
        [1, 2, 3, 5, 7, 8],
        "gptoss20b_big-residual_add",
    ),
    (
        "residual_add",
        (402176, 2880),
        torch.float32,
        [1, 2, 3, 7, 11, 16, 17],
        "gptoss20b_big-residual_add",
    ),
    (
        "embedding_scale",
        (402176, 2880),
        torch.bfloat16,
        [1, 2, 3, 5, 7, 8],
        "gptoss20b_big-embedding_scale",
    ),
    (
        "embedding_scale",
        (402176, 2880),
        torch.float16,
        [1, 2, 3, 5, 7, 8],
        "gptoss20b_big-embedding_scale",
    ),
    (
        "embedding_scale",
        (402176, 2880),
        torch.float32,
        [1, 2, 3, 7, 11, 16, 17],
        "gptoss20b_big-embedding_scale",
    ),
    # mistral_big  (262144, 5120) = 2×131072  bf16/fp16=[1..9]  fp32=[1..19]
    (
        "residual_add",
        (262144, 5120),
        torch.bfloat16,
        [1, 2, 3, 5, 8, 9],
        "mistral_big-residual_add",
    ),
    (
        "residual_add",
        (262144, 5120),
        torch.float16,
        [1, 2, 3, 5, 8, 9],
        "mistral_big-residual_add",
    ),
    (
        "residual_add",
        (262144, 5120),
        torch.float32,
        [1, 2, 3, 8, 13, 18, 19],
        "mistral_big-residual_add",
    ),
    (
        "rmsnorm_gamma_scale",
        (262144, 5120),
        torch.bfloat16,
        [1, 2, 3, 5, 8, 9],
        "mistral_big-rmsnorm_gamma_scale",
    ),
    (
        "rmsnorm_gamma_scale",
        (262144, 5120),
        torch.float16,
        [1, 2, 3, 5, 8, 9],
        "mistral_big-rmsnorm_gamma_scale",
    ),
    (
        "rmsnorm_gamma_scale",
        (262144, 5120),
        torch.float32,
        [1, 2, 3, 8, 13, 18, 19],
        "mistral_big-rmsnorm_gamma_scale",
    ),
    # granite41_big  (200704, 4096) = 2×100352  bf16/fp16=[1..6]  fp32=[1..12]
    (
        "residual_add",
        (200704, 4096),
        torch.bfloat16,
        [1, 2, 4, 5, 6],
        "granite41_big-residual_add",
    ),
    (
        "residual_add",
        (200704, 4096),
        torch.float16,
        [1, 2, 4, 5, 6],
        "granite41_big-residual_add",
    ),
    (
        "residual_add",
        (200704, 4096),
        torch.float32,
        [1, 2, 3, 5, 7, 11, 12],
        "granite41_big-residual_add",
    ),
    (
        "logit_temperature",
        (200704, 4096),
        torch.bfloat16,
        [1, 2, 4, 5, 6],
        "granite41_big-logit_temperature",
    ),
    (
        "logit_temperature",
        (200704, 4096),
        torch.float16,
        [1, 2, 4, 5, 6],
        "granite41_big-logit_temperature",
    ),
    (
        "logit_temperature",
        (200704, 4096),
        torch.float32,
        [1, 2, 3, 5, 7, 11, 12],
        "granite41_big-logit_temperature",
    ),
]
EXPECTED_FAILURES_S2 = {
    "mistral_big-rmsnorm_gamma_scale-bf16-cores1": _L3_ADDEARIMM_XFAIL,
    "mistral_big-rmsnorm_gamma_scale-bf16-cores2": _L3_ADDEARIMM_XFAIL,
    "mistral_big-rmsnorm_gamma_scale-bf16-cores3": _L3_ADDEARIMM_XFAIL,
    "mistral_big-rmsnorm_gamma_scale-fp16-cores1": _L3_ADDEARIMM_XFAIL,
    "mistral_big-rmsnorm_gamma_scale-fp16-cores2": _L3_ADDEARIMM_XFAIL,
    "mistral_big-rmsnorm_gamma_scale-fp16-cores3": _L3_ADDEARIMM_XFAIL,
    "mistral_big-rmsnorm_gamma_scale-fp32-cores1": _L3_ADDEARIMM_XFAIL,
    "mistral_big-rmsnorm_gamma_scale-fp32-cores2": _L3_ADDEARIMM_XFAIL,
    "mistral_big-rmsnorm_gamma_scale-fp32-cores3": _L3_ADDEARIMM_XFAIL,
    "mistral_exact-rmsnorm_gamma_scale-bf16-cores1": _L3_ADDEARIMM_XFAIL,
    "mistral_exact-rmsnorm_gamma_scale-fp16-cores1": _L3_ADDEARIMM_XFAIL,
    "mistral_exact-rmsnorm_gamma_scale-fp32-cores1": _L3_ADDEARIMM_XFAIL,
    "mistral_exact-rmsnorm_gamma_scale-fp32-cores2": _L3_ADDEARIMM_XFAIL,
    "rmsnorm_gamma_scale-120000x4096-bf16-cores1": _L3_ADDEARIMM_XFAIL,
    "rmsnorm_gamma_scale-120000x4096-fp16-cores1": _L3_ADDEARIMM_XFAIL,
    "rmsnorm_gamma_scale-120000x4096-fp32-cores1": _L3_ADDEARIMM_XFAIL,
    "rmsnorm_gamma_scale-140000x5120-bf16-cores1": _L3_ADDEARIMM_XFAIL,
    "rmsnorm_gamma_scale-140000x5120-fp16-cores1": _L3_ADDEARIMM_XFAIL,
    "rmsnorm_gamma_scale-140000x5120-fp32-cores1": _L3_ADDEARIMM_XFAIL,
    "rmsnorm_gamma_scale-140000x5120-fp32-cores2": _L3_ADDEARIMM_XFAIL,
    "embedding_scale-50000x2880-fp32-cores1": _WORK_DIV_XFAIL,
    "embedding_scale-50000x2880-fp32-cores2": _WORK_DIV_XFAIL,
    "gptoss20b_big-embedding_scale-bf16-cores1": _WORK_DIV_XFAIL,
    "gptoss20b_big-embedding_scale-bf16-cores2": _WORK_DIV_XFAIL,
    "gptoss20b_big-embedding_scale-fp32-cores1": _WORK_DIV_XFAIL,
    "gptoss20b_big-embedding_scale-fp32-cores2": _WORK_DIV_XFAIL,
    "gptoss20b_big-embedding_scale-fp32-cores3": _WORK_DIV_XFAIL,
    "gptoss20b_big-residual_add-bf16-cores1": _WORK_DIV_XFAIL,
    "gptoss20b_big-residual_add-bf16-cores2": _WORK_DIV_XFAIL,
    "gptoss20b_big-residual_add-fp16-cores1": _WORK_DIV_XFAIL,
    "gptoss20b_big-residual_add-fp16-cores2": _WORK_DIV_XFAIL,
    "gptoss20b_big-residual_add-fp32-cores1": _WORK_DIV_XFAIL,
    "gptoss20b_big-residual_add-fp32-cores2": _WORK_DIV_XFAIL,
    "gptoss20b_big-residual_add-fp32-cores3": _WORK_DIV_XFAIL,
    "gptoss20b_exact-embedding_scale-bf16-cores1": _WORK_DIV_XFAIL,
    "gptoss20b_exact-embedding_scale-bf16-cores2": _WORK_DIV_XFAIL,
    "gptoss20b_exact-embedding_scale-fp32-cores1": _WORK_DIV_XFAIL,
    "gptoss20b_exact-embedding_scale-fp32-cores2": _WORK_DIV_XFAIL,
    "gptoss20b_exact-residual_add-bf16-cores1": _WORK_DIV_XFAIL,
    "gptoss20b_exact-residual_add-bf16-cores2": _WORK_DIV_XFAIL,
    "gptoss20b_exact-residual_add-fp16-cores1": _WORK_DIV_XFAIL,
    "gptoss20b_exact-residual_add-fp16-cores2": _WORK_DIV_XFAIL,
    "gptoss20b_exact-residual_add-fp32-cores1": _WORK_DIV_XFAIL,
    "gptoss20b_exact-residual_add-fp32-cores2": _WORK_DIV_XFAIL,
    "granite41_big-logit_temperature-bf16-cores1": _WORK_DIV_XFAIL,
    "granite41_big-logit_temperature-fp32-cores1": _WORK_DIV_XFAIL,
    "granite41_big-logit_temperature-fp32-cores2": _WORK_DIV_XFAIL,
    "granite41_big-logit_temperature-fp32-cores3": _WORK_DIV_XFAIL,
    "granite41_big-residual_add-bf16-cores1": _WORK_DIV_XFAIL,
    "granite41_big-residual_add-fp16-cores1": _WORK_DIV_XFAIL,
    "granite41_big-residual_add-fp32-cores1": _WORK_DIV_XFAIL,
    "granite41_big-residual_add-fp32-cores2": _WORK_DIV_XFAIL,
    "granite41_big-residual_add-fp32-cores3": _WORK_DIV_XFAIL,
    "granite41_exact-logit_temperature-bf16-cores1": _WORK_DIV_XFAIL,
    "granite41_exact-logit_temperature-fp32-cores1": _WORK_DIV_XFAIL,
    "granite41_exact-residual_add-bf16-cores1": _WORK_DIV_XFAIL,
    "granite41_exact-residual_add-fp16-cores1": _WORK_DIV_XFAIL,
    "granite41_exact-residual_add-fp32-cores1": _WORK_DIV_XFAIL,
    "llama3_big-logit_temperature-bf16-cores1": _WORK_DIV_XFAIL,
    "llama3_big-logit_temperature-fp32-cores1": _WORK_DIV_XFAIL,
    "llama3_big-logit_temperature-fp32-cores2": _WORK_DIV_XFAIL,
    "llama3_big-logit_temperature-fp32-cores3": _WORK_DIV_XFAIL,
    "llama3_big-residual_add-bf16-cores1": _WORK_DIV_XFAIL,
    "llama3_big-residual_add-fp16-cores1": _WORK_DIV_XFAIL,
    "llama3_big-residual_add-fp32-cores1": _WORK_DIV_XFAIL,
    "llama3_big-residual_add-fp32-cores2": _WORK_DIV_XFAIL,
    "llama3_big-residual_add-fp32-cores3": _WORK_DIV_XFAIL,
    "llama3_exact-logit_temperature-bf16-cores1": _WORK_DIV_XFAIL,
    "llama3_exact-logit_temperature-fp32-cores1": _WORK_DIV_XFAIL,
    "llama3_exact-residual_add-bf16-cores1": _WORK_DIV_XFAIL,
    "llama3_exact-residual_add-fp16-cores1": _WORK_DIV_XFAIL,
    "llama3_exact-residual_add-fp32-cores1": _WORK_DIV_XFAIL,
    "logit_temperature-105000x4096-bf16-cores1": _WORK_DIV_XFAIL,
    "logit_temperature-105000x4096-fp32-cores1": _WORK_DIV_XFAIL,
    "logit_temperature-52000x4096-fp32-cores1": _WORK_DIV_XFAIL,
    "logit_temperature-65x44x49152-fp32-cores1": _WORK_DIV_XFAIL,
    "logit_temperature-75x41x49152-bf16-cores1": _WORK_DIV_XFAIL,
    "logit_temperature-75x41x49152-fp16-cores1": _WORK_DIV_XFAIL,
    "logit_temperature-75x41x49152-fp32-cores1": _WORK_DIV_XFAIL,
    "logit_temperature-75x41x49152-fp32-cores2": _WORK_DIV_XFAIL,
    "logit_temperature-92000x4096-bf16-cores1": _WORK_DIV_XFAIL,
    "logit_temperature-92000x4096-fp16-cores1": _WORK_DIV_XFAIL,
    "mistral_big-residual_add-bf16-cores1": _WORK_DIV_XFAIL,
    "mistral_big-residual_add-bf16-cores2": _WORK_DIV_XFAIL,
    "mistral_big-residual_add-bf16-cores3": _WORK_DIV_XFAIL,
    "mistral_big-residual_add-fp16-cores1": _WORK_DIV_XFAIL,
    "mistral_big-residual_add-fp16-cores2": _WORK_DIV_XFAIL,
    "mistral_big-residual_add-fp16-cores3": _WORK_DIV_XFAIL,
    "mistral_big-residual_add-fp32-cores1": _WORK_DIV_XFAIL,
    "mistral_big-residual_add-fp32-cores2": _WORK_DIV_XFAIL,
    "mistral_big-residual_add-fp32-cores3": _WORK_DIV_XFAIL,
    "mistral_exact-residual_add-bf16-cores1": _WORK_DIV_XFAIL,
    "mistral_exact-residual_add-fp16-cores1": _WORK_DIV_XFAIL,
    "mistral_exact-residual_add-fp32-cores1": _WORK_DIV_XFAIL,
    "mistral_exact-residual_add-fp32-cores2": _WORK_DIV_XFAIL,
    "residual_add-115000x5120-bf16-cores1": _WORK_DIV_XFAIL,
    "residual_add-115000x5120-fp16-cores1": _WORK_DIV_XFAIL,
    "residual_add-125000x5120-bf16-cores1": _WORK_DIV_XFAIL,
    "residual_add-125000x5120-fp16-cores1": _WORK_DIV_XFAIL,
    "residual_add-125000x5120-fp32-cores1": _WORK_DIV_XFAIL,
    "residual_add-125000x5120-fp32-cores2": _WORK_DIV_XFAIL,
    "residual_add-125000x5120-fp32-cores3": _WORK_DIV_XFAIL,
    "residual_add-130000x4096-bf16-cores1": _WORK_DIV_XFAIL,
    "residual_add-130000x4096-fp16-cores1": _WORK_DIV_XFAIL,
    "residual_add-130000x4096-fp32-cores1": _WORK_DIV_XFAIL,
    "residual_add-135000x4096-bf16-cores1": _WORK_DIV_XFAIL,
    "residual_add-135000x4096-fp16-cores1": _WORK_DIV_XFAIL,
    "residual_add-185000x2880-bf16-cores1": _WORK_DIV_XFAIL,
    "residual_add-185000x2880-bf16-cores2": _WORK_DIV_XFAIL,
    "residual_add-185000x2880-fp16-cores1": _WORK_DIV_XFAIL,
    "residual_add-185000x2880-fp16-cores2": _WORK_DIV_XFAIL,
    "residual_add-195000x2880-bf16-cores1": _WORK_DIV_XFAIL,
    "residual_add-195000x2880-bf16-cores2": _WORK_DIV_XFAIL,
    "residual_add-195000x2880-fp16-cores1": _WORK_DIV_XFAIL,
    "residual_add-195000x2880-fp16-cores2": _WORK_DIV_XFAIL,
    "residual_add-195000x2880-fp32-cores1": _WORK_DIV_XFAIL,
    "residual_add-195000x2880-fp32-cores2": _WORK_DIV_XFAIL,
    "residual_add-2750x12x4096-fp32-cores1": _WORK_DIV_XFAIL,
    "residual_add-2800x16x4096-fp32-cores1": _WORK_DIV_XFAIL,
    "residual_add-32x64x1024x2049-fp16-cores1": _WORK_DIV_XFAIL,
    "residual_add-32x64x1024x2049-fp16-cores2": _WORK_DIV_XFAIL,
    "residual_add-47000x4096-fp32-cores1": _WORK_DIV_XFAIL,
    "residual_add-48000x2880-fp32-cores1": _WORK_DIV_XFAIL,
    "residual_add-98000x4096-bf16-cores1": _WORK_DIV_XFAIL,
    "residual_add-98000x4096-fp16-cores1": _WORK_DIV_XFAIL,
    "residual_add-98000x4096-fp32-cores1": _WORK_DIV_XFAIL,
    "rope_cos_mul-10000x32x12x128-bf16-cores1": _WORK_DIV_XFAIL,
    "rope_cos_mul-10000x32x12x128-fp16-cores1": _WORK_DIV_XFAIL,
    "rope_cos_mul-10000x32x12x128-fp32-cores1": _WORK_DIV_XFAIL,
    "rope_cos_mul-28000x32x12x128-bf16-cores1": _WORK_DIV_XFAIL,
    "rope_cos_mul-28000x32x12x128-bf16-cores2": _WORK_DIV_XFAIL,
    "rope_cos_mul-28000x32x12x128-bf16-cores3": _WORK_DIV_XFAIL,
    "rope_cos_mul-28000x32x12x128-fp16-cores1": _WORK_DIV_XFAIL,
    "rope_cos_mul-28000x32x12x128-fp16-cores2": _WORK_DIV_XFAIL,
    "rope_cos_mul-28000x32x12x128-fp16-cores3": _WORK_DIV_XFAIL,
    "rope_cos_mul-28000x32x12x128-fp32-cores1": _WORK_DIV_XFAIL,
    "rope_cos_mul-28000x32x12x128-fp32-cores2": _WORK_DIV_XFAIL,
    "rope_cos_mul-6144x32x12x128-bf16-cores1": _WORK_DIV_XFAIL,
    "rope_cos_mul-6144x32x12x128-fp16-cores1": _WORK_DIV_XFAIL,
    "rope_cos_mul-6144x32x12x128-fp32-cores1": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-bf16-cores1": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-bf16-cores11": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-bf16-cores13": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-bf16-cores2": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-bf16-cores4": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-bf16-cores6": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-bf16-cores9": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-fp16-cores1": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-fp16-cores11": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-fp16-cores13": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-fp16-cores2": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-fp16-cores4": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-fp16-cores6": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-fp16-cores9": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-fp32-cores1": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-fp32-cores12": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-fp32-cores17": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-fp32-cores2": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-fp32-cores22": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-fp32-cores27": _WORK_DIV_XFAIL,
    "rope_sin_mul-30000x32x16x128-fp32-cores7": _WORK_DIV_XFAIL,
    "torch_add-1000x1064x1024-bf16-cores1": _WORK_DIV_XFAIL,
    "torch_add-1000x1064x1024-bf16-cores2": _WORK_DIV_XFAIL,
    "torch_add-1000x1064x1024-bf16-cores3": _WORK_DIV_XFAIL,
    "torch_add-1000x1064x1024-bf16-cores4": _WORK_DIV_XFAIL,
    "torch_add-1000x1064x1024-bf16-cores5": _WORK_DIV_XFAIL,
    "torch_add-1000x1064x1024-bf16-cores6": _WORK_DIV_XFAIL,
    "torch_add-1000x1064x1024-fp16-cores1": _WORK_DIV_XFAIL,
    "torch_add-1000x1064x1024-fp16-cores2": _WORK_DIV_XFAIL,
    "torch_add-1000x1064x1024-fp16-cores3": _WORK_DIV_XFAIL,
    "torch_add-1000x1064x1024-fp16-cores4": _WORK_DIV_XFAIL,
    "torch_add-1000x1064x1024-fp16-cores5": _WORK_DIV_XFAIL,
    "torch_add-1000x1064x1024-fp16-cores6": _WORK_DIV_XFAIL,
    "torch_add-1000x1064x1024-fp32-cores1": _WORK_DIV_XFAIL,
    "torch_add-1000x1064x1024-fp32-cores2": _WORK_DIV_XFAIL,
    "torch_add-1000x1064x1024-fp32-cores5": _WORK_DIV_XFAIL,
    "torch_add-125x1064x1024-fp32-cores1": _WORK_DIV_XFAIL,
    "torch_add-125x16x1064x64-fp32-cores1": _WORK_DIV_XFAIL,
    "torch_add-500x1064x1024-bf16-cores1": _WORK_DIV_XFAIL,
    "torch_add-500x1064x1024-fp16-cores1": _WORK_DIV_XFAIL,
    "torch_add-500x1064x1024-fp32-cores1": _WORK_DIV_XFAIL,
    "torch_add-500x1064x1024-fp32-cores2": _WORK_DIV_XFAIL,
    "torch_add-500x1064x1024-fp32-cores3": _WORK_DIV_XFAIL,
    "torch_add-500x1064x1024-fp32-cores4": _WORK_DIV_XFAIL,
    "torch_add-500x1064x1024-fp32-cores5": _WORK_DIV_XFAIL,
    "torch_add-500x1064x1024-fp32-cores6": _WORK_DIV_XFAIL,
    "torch_add-500x16x1064x64-bf16-cores1": _WORK_DIV_XFAIL,
    "torch_add-500x16x1064x64-fp16-cores1": _WORK_DIV_XFAIL,
    "torch_add-500x16x1064x64-fp32-cores1": _WORK_DIV_XFAIL,
    "torch_add-500x16x1064x64-fp32-cores2": _WORK_DIV_XFAIL,
    "torch_add-500x16x1064x64-fp32-cores3": _WORK_DIV_XFAIL,
    "torch_mul-1000x16x1064x64-bf16-cores1": _WORK_DIV_XFAIL,
    "torch_mul-1000x16x1064x64-bf16-cores2": _WORK_DIV_XFAIL,
    "torch_mul-1000x16x1064x64-bf16-cores3": _WORK_DIV_XFAIL,
    "torch_mul-1000x16x1064x64-fp32-cores1": _WORK_DIV_XFAIL,
    "torch_mul-1000x16x1064x64-fp32-cores10": _WORK_DIV_XFAIL,
    "torch_mul-1000x16x1064x64-fp32-cores12": _WORK_DIV_XFAIL,
    "torch_mul-1000x16x1064x64-fp32-cores15": _WORK_DIV_XFAIL,
    "torch_mul-1000x16x1064x64-fp32-cores2": _WORK_DIV_XFAIL,
    "torch_mul-1000x16x1064x64-fp32-cores5": _WORK_DIV_XFAIL,
    "torch_mul-1000x16x1064x64-fp32-cores7": _WORK_DIV_XFAIL,
    "torch_mul-250x1064x1024-bf16-cores1": _WORK_DIV_XFAIL,
    "torch_mul-250x1064x1024-fp16-cores1": _WORK_DIV_XFAIL,
    "torch_mul-250x1064x1024-fp32-cores1": _WORK_DIV_XFAIL,
    "torch_mul-250x16x1064x64-bf16-cores1": _WORK_DIV_XFAIL,
    "torch_mul-250x16x1064x64-fp16-cores1": _WORK_DIV_XFAIL,
    "torch_mul-250x16x1064x64-fp32-cores1": _WORK_DIV_XFAIL,
    "torch_mul-750x1064x1024-bf16-cores1": _WORK_DIV_XFAIL,
    "torch_mul-750x1064x1024-fp32-cores1": _WORK_DIV_XFAIL,
    "torch_mul-750x1064x1024-fp32-cores2": _WORK_DIV_XFAIL,
    "torch_mul-750x1064x1024-fp32-cores4": _WORK_DIV_XFAIL,
    "torch_mul-750x1064x1024-fp32-cores6": _WORK_DIV_XFAIL,
    "torch_pow-300x1064x1024-bf16-cores1": _WORK_DIV_XFAIL,
    "torch_pow-300x1064x1024-fp16-cores1": _WORK_DIV_XFAIL,
    "torch_pow-300x1064x1024-fp32-cores1": _WORK_DIV_XFAIL,
    "torch_pow-800x1064x1024-bf16-cores1": _WORK_DIV_XFAIL,
    "torch_pow-800x1064x1024-fp32-cores1": _WORK_DIV_XFAIL,
    "torch_pow-800x1064x1024-fp32-cores2": _WORK_DIV_XFAIL,
    "torch_pow-800x1064x1024-fp32-cores4": _WORK_DIV_XFAIL,
    "torch_pow-800x1064x1024-fp32-cores6": _WORK_DIV_XFAIL,
    "torch_rsqrt-400x1064x1024-bf16-cores1": _WORK_DIV_XFAIL,
    "torch_rsqrt-400x1064x1024-fp16-cores1": _WORK_DIV_XFAIL,
    "torch_rsqrt-400x1064x1024-fp32-cores1": _WORK_DIV_XFAIL,
    "torch_rsqrt-400x1064x1024-fp32-cores2": _WORK_DIV_XFAIL,
    "torch_rsqrt-400x1064x1024-fp32-cores3": _WORK_DIV_XFAIL,
    "torch_rsqrt-400x1064x1024-fp32-cores4": _WORK_DIV_XFAIL,
    "torch_rsqrt-400x1064x1024-fp32-cores5": _WORK_DIV_XFAIL,
    "torch_rsqrt-400x1064x1024-fp32-cores6": _WORK_DIV_XFAIL,
    "torch_rsqrt-600x1064x1024-bf16-cores1": _WORK_DIV_XFAIL,
    "torch_rsqrt-600x1064x1024-fp16-cores1": _WORK_DIV_XFAIL,
    "torch_rsqrt-600x1064x1024-fp32-cores1": _WORK_DIV_XFAIL,
    "torch_rsqrt-600x1064x1024-fp32-cores2": _WORK_DIV_XFAIL,
    "torch_rsqrt-600x1064x1024-fp32-cores3": _WORK_DIV_XFAIL,
    "torch_rsqrt-600x1064x1024-fp32-cores5": _WORK_DIV_XFAIL,
    "torch_rsqrt-600x1064x1024-fp32-cores6": _WORK_DIV_XFAIL,
    "torch_rsqrt-600x1064x1024-fp32-cores8": _WORK_DIV_XFAIL,
    "torch_rsqrt-600x1064x1024-fp32-cores9": _WORK_DIV_XFAIL,
}
# =============================================================================
# S6- Plan structural correctness (prime or has very few factors)
# =============================================================================
S6_ROWS = [
    ("rmsnorm_gamma_scale", (1900, 14, 5120), torch.float32, [1]),
    ("rmsnorm_gamma_scale", (2000, 14, 5120), torch.bfloat16, [1]),
    ("rmsnorm_gamma_scale", (2000, 14, 5120), torch.float16, [1]),
    ("rmsnorm_gamma_scale", (2000, 14, 5120), torch.float32, [1, 2]),
    ("sdpa_scale_standalone", (900, 32, 64, 128), torch.bfloat16, [1]),
    ("sdpa_scale_standalone", (900, 32, 64, 128), torch.float16, [1]),
    ("sdpa_scale_standalone", (900, 32, 64, 128), torch.float32, [1, 2, 3]),
    ("sdpa_scale_standalone", (1200, 32, 64, 128), torch.bfloat16, [1, 2]),
    ("sdpa_scale_standalone", (1200, 32, 64, 128), torch.float16, [1, 2]),
    ("sdpa_scale_standalone", (1200, 32, 64, 128), torch.float32, [1, 2, 3, 4]),
    (
        "sdpa_scale_standalone",
        (5500, 32, 64, 128),
        torch.bfloat16,
        [1, 2, 4, 5, 7, 8, 10],
    ),
    (
        "sdpa_scale_standalone",
        (5500, 32, 64, 128),
        torch.float16,
        [1, 2, 4, 5, 7, 8, 10],
    ),
    (
        "sdpa_scale_standalone",
        (5500, 32, 64, 128),
        torch.float32,
        [1, 2, 6, 9, 13, 16, 20],
    ),
    (
        "sdpa_scale_standalone",
        (7000, 32, 64, 128),
        torch.bfloat16,
        [1, 2, 4, 6, 9, 11, 13],
    ),
    (
        "sdpa_scale_standalone",
        (7000, 32, 64, 128),
        torch.float16,
        [1, 2, 4, 6, 9, 11, 13],
    ),
    (
        "sdpa_scale_standalone",
        (7000, 32, 64, 128),
        torch.float32,
        [1, 2, 7, 12, 16, 21, 26],
    ),
    ("torch_neg", (250, 16, 1064, 32), torch.float32, [1]),
    ("torch_neg", (1000, 16, 1064, 32), torch.bfloat16, [1, 2, 3]),
    ("torch_neg", (1000, 16, 1064, 32), torch.float16, [1, 2, 3]),
    ("torch_neg", (1000, 16, 1064, 32), torch.float32, [1, 2, 3, 4, 5, 6, 7]),
]
EXPECTED_FAILURES_S6 = {
    # Bug 3 — issue #3415
    "rmsnorm_gamma_scale-1900x14x5120-fp32-cores1": _WORK_DIV_XFAIL,
    # Bug 2 — issue #3414
    "rmsnorm_gamma_scale-2000x14x5120-fp32-cores1": _L3_ADDEARIMM_XFAIL,
    "rmsnorm_gamma_scale-2000x14x5120-fp32-cores2": _L3_ADDEARIMM_XFAIL,
    # Bug 3 — issue #3415
    "sdpa_scale_standalone-900x32x64x128-fp32-cores1": _WORK_DIV_XFAIL,
    "sdpa_scale_standalone-1200x32x64x128-bf16-cores1": _WORK_DIV_XFAIL,
    "sdpa_scale_standalone-1200x32x64x128-fp16-cores1": _WORK_DIV_XFAIL,
    "sdpa_scale_standalone-1200x32x64x128-fp32-cores1": _WORK_DIV_XFAIL,
    "sdpa_scale_standalone-5500x32x64x128-bf16-cores1": _WORK_DIV_XFAIL,
    "sdpa_scale_standalone-5500x32x64x128-bf16-cores2": _WORK_DIV_XFAIL,
    "sdpa_scale_standalone-5500x32x64x128-fp32-cores1": _WORK_DIV_XFAIL,
    "sdpa_scale_standalone-5500x32x64x128-fp32-cores2": _WORK_DIV_XFAIL,
    "sdpa_scale_standalone-7000x32x64x128-bf16-cores1": _WORK_DIV_XFAIL,
    "sdpa_scale_standalone-7000x32x64x128-bf16-cores2": _WORK_DIV_XFAIL,
    "sdpa_scale_standalone-7000x32x64x128-fp32-cores1": _WORK_DIV_XFAIL,
    "sdpa_scale_standalone-7000x32x64x128-fp32-cores2": _WORK_DIV_XFAIL,
    "torch_neg-250x16x1064x32-fp32-cores1": _WORK_DIV_XFAIL,
    "torch_neg-1000x16x1064x32-bf16-cores1": _WORK_DIV_XFAIL,
    "torch_neg-1000x16x1064x32-bf16-cores2": _WORK_DIV_XFAIL,
    "torch_neg-1000x16x1064x32-bf16-cores3": _WORK_DIV_XFAIL,
    "torch_neg-1000x16x1064x32-fp16-cores1": _WORK_DIV_XFAIL,
    "torch_neg-1000x16x1064x32-fp16-cores2": _WORK_DIV_XFAIL,
    "torch_neg-1000x16x1064x32-fp16-cores3": _WORK_DIV_XFAIL,
    "torch_neg-1000x16x1064x32-fp32-cores1": _WORK_DIV_XFAIL,
    "torch_neg-1000x16x1064x32-fp32-cores2": _WORK_DIV_XFAIL,
    "torch_neg-1000x16x1064x32-fp32-cores3": _WORK_DIV_XFAIL,
}
# =============================================================================
# S9 -- Aggregate span limit (total bytes across all cores > 32*256 MiB).
# =============================================================================
S9_ROWS = [
    ("logit_temperature", (270000, 16384), torch.bfloat16, [32], "vocab_stress_2d"),
    ("swiglu_silu", (32, 9728, 14336), torch.bfloat16, [32], "swiglu_stress_3d"),
    (
        "rmsnorm_gamma_scale",
        (16, 18000, 16384),
        torch.bfloat16,
        [32],
        "activ_stress_3d",
    ),
    ("rope_cos_mul", (32, 48, 2048, 2048), torch.bfloat16, [32], "rope_stress_4d"),
]
EXPECTED_FAILURES_S9 = {
    "vocab_stress_2d-bf16-cores32": _PLANNER_XFAIL,  # Bug 4 — issue #3417
}
# =============================================================================
# S12 -- Auto-vs-manual E2E equivalence & codegen.
# =============================================================================
_S12_SHAPE = (49159, 4096)
_S12_DTYPE = torch.bfloat16
_S12_CORES = 1
_S12_SPLIT_COUNT = 2

EXPECTED_FAILURES_S12 = {}


def _s12_mark(method_name):
    if method_name not in EXPECTED_FAILURES_S12:
        return None
    action, reason = EXPECTED_FAILURES_S12[method_name]
    return (
        pytest.mark.xfail(reason=reason)
        if action == "xfail"
        else pytest.mark.skip(reason=reason)
    )


# =============================================================================
# TEST CLASS
# =============================================================================
class TestOps:
    """Plain pytest class for all span_overflow_hint_analysis correctness tests."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)
        torch._dynamo.reset_code_caches()
        torch._inductor.codecache.FxGraphCache.clear()

    def teardown_method(self):
        """Force GC + dynamo reset after every test to free device memory."""
        gc.collect()
        torch._dynamo.reset()
        os.environ.pop("SENCORES", None)
        spyre_config.sencores = 32

    @pytest.fixture(autouse=True)
    def env_span_overflow(self):
        os.environ.pop("SENCORES", None)
        os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        spyre_config.sencores = 32
        with spyre_config.patch({"ignore_span_overflow_hints": False}):
            yield
        os.environ.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
        os.environ.pop("SENCORES", None)
        spyre_config.sencores = 32

    def compare_with_cpu(self, *args, **kwargs):
        kwargs.setdefault("run_eager", False)
        kwargs.setdefault("cpu_compile", True)
        return utils_inductor.compare_with_cpu(*args, **kwargs)

    def run_span_overflow_test(self, fn, x, y=None, cores=None):
        if cores is not None:
            spyre_config.sencores = cores
        input_shapes = [t.shape for t in (x, y) if isinstance(t, torch.Tensor)]
        expected_shape = torch.broadcast_shapes(*input_shapes)
        device_args = [t.to("spyre") for t in (x, y) if t is not None]
        torch._dynamo.reset()
        spyre_out = torch.compile(fn, dynamic=False)(*device_args)
        assert spyre_out.shape == expected_shape, (
            f"Spyre tiling changed tensor shape -- output {spyre_out.shape}, expected {expected_shape}"
        )
        atol, rtol = _tol(x.dtype)
        if y is not None:
            self.compare_with_cpu(
                fn, x, y, atol=atol, rtol=rtol, target=spyre_out.cpu()
            )
        else:
            self.compare_with_cpu(fn, x, atol=atol, rtol=rtol, target=spyre_out.cpu())

    # -------------------------------------------------------------------------
    # S0 -- Baseline op-formula correctness
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "op_key, shape, dtype, cores",
        _expand_rows(S0_ROWS, EXPECTED_FAILURES_S0),
    )
    def test_S0_baseline_correctness(self, op_key, shape, dtype, cores):
        fn, a, b = _build_inputs(op_key, shape, dtype)
        self.run_span_overflow_test(fn, a, b, cores)

    # -------------------------------------------------------------------------
    # S1 -- Pass-through / negative result
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "op_key, shape, dtype, cores",
        _expand_rows(S1_ROWS, EXPECTED_FAILURES_S1),
    )
    def test_S1_pass_through(self, op_key, shape, dtype, cores):
        fn, a, b = _build_inputs(op_key, shape, dtype)
        self.run_span_overflow_test(fn, a, b, cores)

    # -------------------------------------------------------------------------
    # S4 -- Boundary: strict >
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "op_key, shape, dtype, cores",
        _expand_rows(S4_ROWS, EXPECTED_FAILURES_S4),
    )
    def test_S4_boundary(self, op_key, shape, dtype, cores):
        fn, a, b = _build_inputs(op_key, shape, dtype)
        self.run_span_overflow_test(fn, a, b, cores)

    # -------------------------------------------------------------------------
    # S2 - per-core span overflow.
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "op_key, shape, dtype, cores",
        _expand_rows(S2_ROWS, EXPECTED_FAILURES_S2),
    )
    def test_S2_trigger_a_vocab_width_op(self, op_key, shape, dtype, cores):
        fn, a, b = _build_inputs(op_key, shape, dtype)
        self.run_span_overflow_test(fn, a, b, cores)

    # -------------------------------------------------------------------------
    # S6 -- Plan structural correctness.
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "op_key, shape, dtype, cores",
        _expand_rows(S6_ROWS, EXPECTED_FAILURES_S6),
    )
    def test_S6_swiglu_plan_check(self, op_key, shape, dtype, cores):
        fn, a, b = _build_inputs(op_key, shape, dtype)
        self.run_span_overflow_test(fn, a, b, cores)

    # -------------------------------------------------------------------------
    # S9 -- Aggregate span limit / production-scale stress
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "op_key, shape, dtype, cores",
        _expand_rows(S9_ROWS, EXPECTED_FAILURES_S9),
    )
    def test_S9_production_stress(self, op_key, shape, dtype, cores):
        fn, a, b = _build_inputs(op_key, shape, dtype)
        self.run_span_overflow_test(fn, a, b, cores)


# =============================================================================
# S12 -- Coarse-tile stamping, codegen & E2E equivalence.
# =============================================================================
class TestS12CoarseTileStampingAndCodegen:
    """S12 -- Auto-vs-Manual E2E Equivalence & Codegen"""

    def setup_method(self):
        torch.manual_seed(0xAFFE)
        torch._dynamo.reset_code_caches()
        torch._inductor.codecache.FxGraphCache.clear()

    def teardown_method(self):
        gc.collect()
        torch._dynamo.reset()
        os.environ.pop("SENCORES", None)
        spyre_config.sencores = 32

    @pytest.fixture(autouse=True)
    def env_span_overflow(self):
        os.environ.pop("SENCORES", None)
        os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        spyre_config.sencores = 32
        with spyre_config.patch({"ignore_span_overflow_hints": False}):
            yield
        os.environ.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
        os.environ.pop("SENCORES", None)
        spyre_config.sencores = 32

    def test_auto_vs_manual_e2e_output_correctness(self):
        from torch_spyre._inductor import spyre_hint

        spyre_config.sencores = _S12_CORES

        def _manual_hint_add(x, y):
            with spyre_hint(num_tiles_per_dim={"SO_ROW": _S12_SPLIT_COUNT}):
                return x + y

        x_auto = _mk(_S12_SHAPE, _S12_DTYPE).to("spyre")
        y_auto = _mk(_S12_SHAPE, _S12_DTYPE).to("spyre")
        torch._dynamo.reset()
        auto_out = torch.compile(_residual_add, dynamic=False)(x_auto, y_auto)
        assert auto_out.shape == torch.Size(_S12_SHAPE)
        _atol, _rtol = _tol(_S12_DTYPE)
        torch.testing.assert_close(
            auto_out.cpu(),
            _residual_add(x_auto.cpu(), y_auto.cpu()),
            atol=_atol,
            rtol=_rtol,
        )
        x_manual = _mk(_S12_SHAPE, _S12_DTYPE).to("spyre")
        y_manual = _mk(_S12_SHAPE, _S12_DTYPE).to("spyre")
        _pnd.declare_tensor_dim("SO_ROW", _S12_SHAPE[0])
        _pnd.declare_tensor_dim("SO_COL", _S12_SHAPE[1])
        _pnd.name_tensor_dims(x_manual, ["SO_ROW", "SO_COL"])
        _pnd.name_tensor_dims(y_manual, ["SO_ROW", "SO_COL"])
        torch._dynamo.reset()
        manual_out = torch.compile(_manual_hint_add, dynamic=False)(
            x_manual, y_manual
        ).cpu()
        assert manual_out.shape == torch.Size(_S12_SHAPE)
        torch.testing.assert_close(
            manual_out,
            _residual_add(x_manual.cpu(), y_manual.cpu()),
            atol=_atol,
            rtol=_rtol,
        )

    @spyre_config.patch(
        {
            "sencores": _S12_CORES,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_codegen_contains_auto_span_overflow_loop_spec(self):
        x = _mk(_S12_SHAPE, _S12_DTYPE).to("spyre")
        y = _mk(_S12_SHAPE, _S12_DTYPE).to("spyre")
        cfn = torch.compile(_residual_add, dynamic=False)
        result, source_codes = run_and_get_code(cfn, x, y)
        spyre_output = result.cpu()
        cpu_output = x.cpu() + y.cpu()
        assert result.shape == torch.Size(_S12_SHAPE)
        _atol, _rtol = _tol(_S12_DTYPE)
        torch.testing.assert_close(spyre_output, cpu_output, atol=_atol, rtol=_rtol)
        assert source_codes
        src = source_codes[0]
        assert "LoopSpec(" in src
        assert f"sympify('{_S12_SPLIT_COUNT}')" in src

    @spyre_config.patch(
        {
            "sencores": _S12_CORES,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    )
    def test_auto_span_overflow_matches_equivalent_spyre_hint_loop_spec(self):
        from torch_spyre._inductor import spyre_hint

        x = _mk(_S12_SHAPE, _S12_DTYPE).to("spyre")
        y = _mk(_S12_SHAPE, _S12_DTYPE).to("spyre")

        def manual_hint_fn(x, y):
            with spyre_hint(num_tiles_per_dim={"SO_ROW": _S12_SPLIT_COUNT}):
                return x + y

        _pnd.declare_tensor_dim("SO_ROW", _S12_SHAPE[0])
        _pnd.declare_tensor_dim("SO_COL", _S12_SHAPE[1])
        _pnd.name_tensor_dims(x, ["SO_ROW", "SO_COL"])
        _pnd.name_tensor_dims(y, ["SO_ROW", "SO_COL"])
        auto_result, auto_sources = run_and_get_code(
            torch.compile(_residual_add, dynamic=False), x, y
        )
        manual_result, manual_sources = run_and_get_code(
            torch.compile(manual_hint_fn, dynamic=False), x, y
        )
        assert auto_result.shape == torch.Size(_S12_SHAPE)
        assert manual_result.shape == torch.Size(_S12_SHAPE)
        ref = x.cpu() + y.cpu()
        _atol, _rtol = _tol(_S12_DTYPE)
        torch.testing.assert_close(auto_result.cpu(), ref, atol=_atol, rtol=_rtol)
        torch.testing.assert_close(manual_result.cpu(), ref, atol=_atol, rtol=_rtol)
        auto_src = auto_sources[0]
        manual_src = manual_sources[0]
        assert auto_src.count("LoopSpec(") == manual_src.count("LoopSpec(")
        assert auto_src.count(f"sympify('{_S12_SPLIT_COUNT}')") == 1
        assert manual_src.count(f"sympify('{_S12_SPLIT_COUNT}')") == 1
        assert "op='add'" in auto_src
        assert "op='add'" in manual_src


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
