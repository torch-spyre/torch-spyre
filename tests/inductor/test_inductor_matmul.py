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
import pytest
import torch
from utils_inductor import compare_with_cpu


def _compare_modes(execution_mode, fn, *args, atol=0.1, rtol=0.1):
    compare_with_cpu(
        fn,
        *args,
        atol=atol,
        rtol=rtol,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


def _tol(dtype):
    return (1e-3, 1e-2) if dtype == torch.float16 else (1e-4, 1e-3)


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestMatmulOps:
    # ── Scenario 1 — Degenerate 1×1 [NEW] ────────────────────────────────────
    # Not in upstream op_db for custom backends. Tests Spyre single-element
    # tiling path.
    @pytest.mark.parametrize(
        "fn,a,b",
        [
            (
                torch.mm,
                torch.tensor([[3.0]], dtype=torch.float16),
                torch.tensor([[4.0]], dtype=torch.float16),
            ),
            (
                torch.bmm,
                torch.tensor([[[3.0]]], dtype=torch.float16),
                torch.tensor([[[4.0]]], dtype=torch.float16),
            ),
            (
                torch.matmul,
                torch.tensor([[5.0]], dtype=torch.float16),
                torch.tensor([[6.0]], dtype=torch.float16),
            ),
        ],
        ids=["mm", "bmm", "matmul"],
    )
    def test_one_by_one(self, execution_mode, fn, a, b):
        atol, rtol = _tol(torch.float16)
        _compare_modes(execution_mode, fn, a, b, atol=atol, rtol=rtol)

    # ── Scenario 3 — Identity matrix correctness [NEW] ───────────────────────
    # Upstream has this for CPU/CUDA but not via compare_with_cpu on a custom
    # backend. Tests Spyre tile engine with non-uniform stride inputs.
    @pytest.mark.parametrize(
        "fn,left,batched",
        [
            (torch.mm, False, False),
            (torch.mm, True, False),
            (torch.bmm, False, True),
        ],
        ids=["mm_right", "mm_left", "bmm_batched"],
    )
    def test_identity(self, execution_mode, fn, left, batched):
        torch.manual_seed(0)
        a = (
            torch.randn(4, 8, 8, dtype=torch.float16)
            if batched
            else torch.randn(8, 8, dtype=torch.float16)
        )
        eye = (
            torch.eye(8, dtype=torch.float16)
            .unsqueeze(0)
            .expand(4, -1, -1)
            .contiguous()
            if batched
            else torch.eye(8, dtype=torch.float16)
        )
        atol, rtol = _tol(torch.float16)
        _compare_modes(
            execution_mode, fn, *(eye, a) if left else (a, eye), atol=atol, rtol=rtol
        )
