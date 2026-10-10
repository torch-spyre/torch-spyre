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

"""Fused downstream operations after gather: unary (neg/sigmoid/relu/tanh/exp/log/sqrt/cos), scalar arithmetic, reductions (mean/sum), named tensor dimensions, and multi-core SENCORES variants."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_inductor import DEVICE, cached_randn  # noqa: E402
from conftest import _xfail_existing, compare_mode  # noqa: E402

_ATOL_F16 = 1e-2
_ATOL_BF16 = 2e-2
_ATOL_F32 = 1e-5

_INDEX_EAGER = (1219, "aten::index.Tensor_out is not registered on Spyre.")
_CLIP_FP32 = (
    4720,
    "Compile F.normalize / clamp_min / clip on IEEE_FP32 is unsupported.",
)


class TestGatherFusedDownstreamOperations:
    """Downstream ops fused with gather: unary (neg/sigmoid/relu/tanh/exp/log/sqrt/cos), scalar mul/div/sub, mean/sum reductions, named embedding dims, and compile cache verification."""

    def setup_method(self):
        torch.manual_seed(0xAFFE)

    @pytest.fixture(autouse=True)
    def env_base(self, patch_sencores):
        yield
        os.environ.pop("SPYRE_INDUCTOR_ENABLE_FUSION", None)

    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "op,diff_key",
        [
            (torch.neg, "gds01"),
            (torch.sigmoid, "gds02"),
            (torch.relu, "gds03"),
        ],
    )
    def test_gather_3d_unary(self, execution_mode, op, diff_key):
        """3D gather on (8,32,128) + unary op co-scheduled."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((8, 32, 128), differentiation=diff_key, dtype=torch.float16)
        idx = torch.randint(0, 8, (4,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: op(x[i]),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_sqrt_bfloat16(self, execution_mode):
        """bfloat16 gather + sqrt downstream; abs input for valid sqrt."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn(
            (32, 128), differentiation="gds04", dtype=torch.bfloat16, abs=True
        )
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.sqrt(x[i]),
            x,
            idx,
            atol=_ATOL_BF16,
            rtol=_ATOL_BF16,
        )

    def test_gather_exp_float32(self, execution_mode):
        """float32 gather + exp downstream; no fp16 precision loss."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4720
        _xfail_existing(execution_mode, eager=_INDEX_EAGER, compiled=_CLIP_FP32)
        x = cached_randn((32, 128), differentiation="gds05", dtype=torch.float32)
        x = x.abs()
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.exp(x[i].clamp(max=10.0)),
            x,
            idx,
            atol=_ATOL_F32,
            rtol=_ATOL_F32,
        )

    def test_gather_triple_unary(self, execution_mode):
        """Triple-chained exp → tanh → sigmoid after gather."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 256), differentiation="gds06", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.sigmoid(torch.tanh(torch.exp(x[i].clamp(max=5.0)))),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_scalar_sub(self, execution_mode):
        """Scalar subtraction (- 0.5) after gather."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 128), differentiation="gds07", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x[i] - 0.5,
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_scalar_div(self, execution_mode):
        """Scalar divide (/ 8.0) — attention head scale."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 256), differentiation="gds08", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x[i] / 8.0,
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_mean_reduction(self, execution_mode):
        """mean(dim=1) after gather; output collapses to (16,)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 128), differentiation="gds09", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x[i].mean(dim=1),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_sum_dim0(self, execution_mode):
        """sum(dim=0) after gather; output shape (128,)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 128), differentiation="gds10", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x[i].sum(dim=0),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_cos_cpu_fallback(self, execution_mode):
        """cos has no Spyre kernel; gather on Spyre, cos falls back to CPU."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 64), differentiation="gds11", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.cos(x[i].float()).half(),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_log_cpu_fallback(self, execution_mode):
        """log unsupported on Spyre; gather on Spyre, log on CPU."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/4720
        _xfail_existing(execution_mode, eager=_INDEX_EAGER, compiled=_CLIP_FP32)
        x = cached_randn(
            (32, 64), differentiation="gds12", dtype=torch.float16, abs=True
        )
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.log(x[i].float().clamp(min=1e-6)).half(),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_softmax(self, execution_mode):
        """gather + softmax(dim=-1); rows sum to 1.0."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 128), differentiation="gds13", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.softmax(x[i].float(), dim=-1).half(),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_abs_then_sum(self, execution_mode):
        """abs → sum(dim=1) two-stage downstream chain."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 256), differentiation="gds14", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.abs(x[i]).sum(dim=1),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_scalar_mul_3d(self, execution_mode):
        """Scalar multiply (* 2.0) on 3D gathered output (4,16,64)."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((8, 16, 64), differentiation="gds15", dtype=torch.float16)
        idx = torch.randint(0, 8, (4,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: x[i] * 2.0,
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------

    def _name_dims(self, tensor, dim_map):
        import torch_spyre._inductor.wsr.propagate_named_dims as _pnd

        for name, size in dim_map.items():
            _pnd.declare_tensor_dim(name, size)
        _pnd.name_tensor_dims(tensor, list(dim_map.keys()))

    def test_named_embedding(self, execution_mode):
        """Embedding table naming: vocab, dim axes labeled."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        w = cached_randn((32000, 128), differentiation="ndm04", dtype=torch.float16)
        idx = torch.randint(0, 32000, (32,), dtype=torch.int64)
        self._name_dims(w, {"vocab": 32000, "dim": 128})
        self._name_dims(idx, {"seq": 32})
        compare_mode(
            execution_mode, lambda w, i: w[i], w, idx, atol=_ATOL_F16, rtol=_ATOL_F16
        )

    def test_named_with_downstream(self, execution_mode):
        """Named dims preserved through gather → exp → sum chain."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((32, 128), differentiation="ndm10", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        self._name_dims(x, {"M": 32, "N": 128})
        self._name_dims(idx, {"P": 16})
        compare_mode(
            execution_mode,
            lambda x, i: torch.exp(x[i]).sum(dim=1),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    # ------------------------------------------------------------------

    def test_fusion_disabled(self, execution_mode):
        """ENABLE_FUSION=0 — gather and downstream as separate kernels."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        os.environ["SPYRE_INDUCTOR_ENABLE_FUSION"] = "0"
        x = cached_randn((32, 256), differentiation="gem10", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, i: torch.exp(x[i]),
            x,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_compile_cache_hit_with_downstream(self, execution_mode):
        """Gather + downstream fused op compiled twice; second call uses cached graph."""
        x = cached_randn((32, 256), differentiation="gem12", dtype=torch.float16)
        idx = torch.randint(0, 32, (16,), dtype=torch.int64)
        fn = torch.compile(lambda x, i: torch.relu(x[i]))
        r1 = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
        r2 = fn(x.to(DEVICE), idx.to(DEVICE)).cpu()
        torch.testing.assert_close(r1, r2, atol=0, rtol=0)

    def test_gather_then_add_broadcast_bias(self, execution_mode):
        """Gather rows then add broadcast bias; common transformer pre-norm pattern."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((64, 128), differentiation="ds_bias01", dtype=torch.float16)
        bias = cached_randn((128,), differentiation="ds_bias01b", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, b, i: x[i].add(b),
            x,
            bias,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )

    def test_gather_then_mul_and_add_residual(self, execution_mode):
        """Gather rows, scale by gate, add residual; MoE output accumulation pattern."""
        # TODO: ISSUE https://github.com/torch-spyre/torch-spyre/issues/1219
        _xfail_existing(execution_mode, eager=_INDEX_EAGER)
        x = cached_randn((64, 128), differentiation="ds_gate01", dtype=torch.float16)
        residual = cached_randn(
            (16, 128), differentiation="ds_gate01r", dtype=torch.float16
        )
        gate = cached_randn((16, 1), differentiation="ds_gate01g", dtype=torch.float16)
        idx = torch.randint(0, 64, (16,), dtype=torch.int64)
        compare_mode(
            execution_mode,
            lambda x, r, g, i: x[i] * g + r,
            x,
            residual,
            gate,
            idx,
            atol=_ATOL_F16,
            rtol=_ATOL_F16,
        )
