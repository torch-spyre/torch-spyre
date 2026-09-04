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

# Bug 2: dxp_standalone SIGABRT when tiling large broadcast-weight ops (stride/offset value out of range)
#        https://github.com/torch-spyre/torch-spyre/issues/3414
_BROADCAST_WEIGHT_XFAIL = (
    "xfail",
    "dxp_standalone SIGABRT when tiling large broadcast-weight ops (stride/offset value out of range) (issue #3414)",
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


# =============================================================================
# S12 -- Auto-vs-manual E2E equivalence & codegen.
# Real Granite-3.3-8B canonical shape (49159, 4096) bf16 at cores=1.
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
