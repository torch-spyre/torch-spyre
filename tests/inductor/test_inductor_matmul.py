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
    # bmm_unit_batch_* cases: issue #4155 regression (unit-batch, stick-
    # aligned direct bmm -- sets SHARED_WEIGHT_UNIT_BMM_INFO_KEY).
    @pytest.mark.parametrize(
        "fn,left,batch,size",
        [
            (torch.mm, False, 0, 8),
            (torch.mm, True, 0, 8),
            (torch.bmm, False, 4, 8),
            (torch.bmm, False, 1, 1024),
            (torch.bmm, False, 1, 2048),
        ],
        ids=[
            "mm_right",
            "mm_left",
            "bmm_batched",
            "bmm_unit_batch_1024",
            "bmm_unit_batch_2048",
        ],
    )
    def test_identity(self, execution_mode, fn, left, batch, size):
        torch.manual_seed(0)
        a = (
            torch.randn(batch, size, size, dtype=torch.float16)
            if batch
            else torch.randn(size, size, dtype=torch.float16)
        )
        eye = (
            torch.eye(size, dtype=torch.float16)
            .unsqueeze(0)
            .expand(batch, -1, -1)
            .contiguous()
            if batch
            else torch.eye(size, dtype=torch.float16)
        )
        atol, rtol = _tol(torch.float16)
        _compare_modes(
            execution_mode, fn, *(eye, a) if left else (a, eye), atol=atol, rtol=rtol
        )

    # ── Shared-weight unit-BMM regression [issue #4155] ──────────────────────
    # reversed()->nonstick bug swapped M/K device axes for unit-batch matmul
    # with a shared weight. Covers both trigger sites (view->mm->view via
    # torch.matmul, direct aten.bmm via torch.bmm); the one-hot marker row
    # makes a reintroduced swap fail under compare_with_cpu.
    @pytest.mark.parametrize("site", ["view_mm_view", "direct_bmm"])
    @pytest.mark.parametrize(
        "M,K,N",
        [
            (256, 1024, 1024),
            (512, 4096, 12800),
        ],
        ids=["issue_4155", "sendnn_like"],
    )
    def test_shared_weight_unit_bmm_marker_row(
        self, execution_mode, M, K, N, site
    ):
        fn = torch.matmul if site == "view_mm_view" else torch.bmm
        torch.manual_seed(hash((M, K, N, site)) & 0xFFFFFFFF)
        row, col = M // 4, K // 2
        x = torch.zeros(1, M, K, dtype=torch.float16)
        x[0, row, col] = 1.0
        w = (
            torch.randn(1, K, N, dtype=torch.float16)
            if site == "direct_bmm"
            else torch.randn(K, N, dtype=torch.float16)
        )
        atol, rtol = _tol(torch.float16)
        _compare_modes(execution_mode, fn, x, w, atol=atol, rtol=rtol)


# ── Sanity-check safety net [issue #4155, Change B] ─────────────────────────
# create_op_spec's post-_preserve sync only touches args reported as
# rewritten; every other arg still hits the original strict check. These
# corrupt a non-bmm op's coordinates without reporting it, proving the
# check still fires outside the carve-out. "add"/"sum" hit create_op_spec's
# two distinct call sites (pointwise vs. reduction).


def _install_coordinate_corruption(monkeypatch, target_op):
    """Corrupt args[0].device_coordinates for target_op without reporting
    it as rewritten, simulating a bug elsewhere in the pipeline.
    """
    import torch_spyre._inductor.spyre_kernel as _sk

    orig = _sk._preserve_shared_weight_unit_bmm_dim
    state = {"corrupted": False}

    def _corrupting(op, it_space, args, op_info):
        it_space_out, rewritten = orig(op, it_space, args, op_info)
        if not state["corrupted"] and op == target_op and args and not rewritten:
            arg0 = args[0]
            if arg0.device_coordinates:
                arg0.device_coordinates[0] = arg0.device_coordinates[0] + 999999
                state["corrupted"] = True
        return it_space_out, rewritten

    monkeypatch.setattr(_sk, "_preserve_shared_weight_unit_bmm_dim", _corrupting)
    return state


def _matmul_then_add(x, w):
    return torch.matmul(x, w) + 1.0


def _matmul_then_sum(x, w):
    return torch.matmul(x, w).sum(dim=-1)


@pytest.mark.parametrize(
    "target_op,fn",
    [
        ("add", _matmul_then_add),
        ("sum", _matmul_then_sum),
    ],
    ids=["add", "sum_reduction"],
)
def test_shared_weight_unit_bmm_sanity_check_fires_outside_carveout(
    monkeypatch, target_op, fn
):
    state = _install_coordinate_corruption(monkeypatch, target_op)
    torch.manual_seed(0)
    x = torch.randn(1, 64, 128, dtype=torch.float16).to("spyre")
    w = torch.randn(128, 128, dtype=torch.float16).to("spyre")

    torch._dynamo.reset()
    with pytest.raises(
        RuntimeError, match="alignment input collection disagrees with tensor codegen"
    ):
        torch.compile(fn, dynamic=False)(x, w)

    assert state["corrupted"], (
        f"corruption hook never fired for op={target_op!r} -- this run "
        "does not prove the safety net still works for it"
    )
