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
import utils_inductor
from torch_spyre._inductor import config as spyre_config

_TOLERANCES = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1e-2, 1e-2),
    torch.float32: (1e-5, 1e-5),
}


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
    """torch.add(x, y) -- binary elementwise."""
    return torch.add(x, y)


def _torch_mul(x, y):
    """torch.mul(x, y) -- binary elementwise."""
    return torch.mul(x, y)


def _torch_pow(x):
    """torch.pow(x, 2.0) -- unary elementwise."""
    return torch.pow(x, 2.0)


def _torch_rsqrt(x):
    """torch.rsqrt(x) -- unary elementwise."""
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


def _expand_rows(rows):
    out = []
    for entry in rows:
        op_key, shape, dtypes, cores_list = entry[:4]
        prefix = entry[4] if len(entry) > 4 else f"{op_key}-{_shape_str(shape)}"
        dtype_list = dtypes if isinstance(dtypes, (list, tuple)) else [dtypes]
        for dtype in dtype_list:
            for c in cores_list:
                pid = f"{prefix}-{_DTYPE_SHORT[dtype]}-cores{c}"
                out.append(pytest.param(op_key, shape, dtype, c, id=pid))
    return out


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


# =============================================================================
# TEST CLASS
# =============================================================================
class TestOps:
    """S9 aggregate span limit / production-scale stress tests."""

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
    # S9 -- Aggregate span limit / production-scale stress
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("op_key, shape, dtype, cores", _expand_rows(S9_ROWS))
    def test_S9_production_stress(self, op_key, shape, dtype, cores):
        if (
            op_key == "logit_temperature"
            and shape == (270000, 16384)
            and dtype == torch.bfloat16
            and cores == 32
        ):
            pytest.skip(
                "span-overflow planner exhausts 512 split candidates; "
                "Unsupported raised (issue #3417)"
            )
        fn, a, b = _build_inputs(op_key, shape, dtype)
        self.run_span_overflow_test(fn, a, b, cores)
