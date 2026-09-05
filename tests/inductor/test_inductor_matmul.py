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
from torch._inductor.utils import run_and_get_code
from torch.spyre import SpyreTensorLayout
from utils_inductor import compare_with_cpu

from torch_spyre import require_layout


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


class TestRequireLayout:
    @pytest.mark.parametrize("dtype", [torch.bool, torch.float64, torch.int64])
    def test_rejects_unsupported_dtype(self, dtype):
        x = torch.ones(1, dtype=dtype)

        with pytest.raises(ValueError, match="require_layout supports"):
            require_layout(x, [], [])

    def test_bmm_emits_requested_output_layout(self):
        x = torch.randn(1, 128, 256, dtype=torch.float16)
        weight = torch.randn(256, 512, dtype=torch.float16)
        target = SpyreTensorLayout(
            [1, 128, 512], [65536, 512, 1], torch.float16, [1, 0, 2]
        )

        device_size = list(target.device_size)
        stride_map = list(target.stride_map)

        def fn(a, b):
            return require_layout(torch.matmul(a, b), device_size, stride_map)

        result, source_codes = run_and_get_code(
            torch.compile(fn), x.to("spyre"), weight.to("spyre")
        )
        torch.testing.assert_close(
            result.cpu(), torch.matmul(x, weight), atol=0.2, rtol=0.1
        )
        self_layout = result.device_tensor_layout()
        assert self_layout is not None
        assert self_layout == target
        assert "device_size=[1, 8, 128, 64]" in source_codes[0]

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_pointwise_emits_requested_output_layout(self, dtype):
        x = torch.randn(2, 128, dtype=dtype)
        x_spyre = x.to("spyre")
        target = x_spyre.device_tensor_layout()

        result, _ = run_and_get_code(
            torch.compile(
                lambda a: require_layout(
                    a + 1, list(target.device_size), list(target.stride_map)
                )
            ),
            x_spyre,
        )

        assert result.device_tensor_layout() == target
        torch.testing.assert_close(result.cpu(), x + 1, atol=0.2, rtol=0.1)

    def test_view_chain_emits_requested_bmm_layout(self):
        x = torch.randn(1, 128, 256, dtype=torch.float16)
        weight = torch.randn(256, 512, dtype=torch.float16)
        target = SpyreTensorLayout(
            [1, 128, 512], [65536, 512, 1], torch.float16, [1, 0, 2]
        )

        device_size = list(target.device_size)
        stride_map = list(target.stride_map)

        def fn(a, b):
            return require_layout(
                torch.matmul(a, b).view(1, 128, 512), device_size, stride_map
            )

        result = torch.compile(fn)(x.to("spyre"), weight.to("spyre"))

        assert result.device_tensor_layout() == target

    def test_rejects_eager_execution(self):
        x = torch.randn(2, 128, dtype=torch.float16).to("spyre")

        with pytest.raises(RuntimeError, match="only inside torch.compile"):
            require_layout(x, [2, 2, 64], [128, 64, 1])

    def test_conflicting_requests_fail(self):
        x = torch.randn(1, 128, 256, dtype=torch.float16)
        weight = torch.randn(256, 512, dtype=torch.float16)

        def fn(a, b):
            output = torch.matmul(a, b)
            first = require_layout(output, [1, 8, 128, 64], [65536, 64, 512, 1])
            second = require_layout(output, [1, 8, 128, 64], [65536, 512, 64, 1])
            return first + second

        with pytest.raises(Exception, match="conflicting require_layout"):
            torch.compile(fn)(x.to("spyre"), weight.to("spyre"))

    def test_equal_requests_on_fused_producers(self):
        x = torch.randn(2, 128, dtype=torch.float16).to("spyre")
        target = x.device_tensor_layout()
        device_size = list(target.device_size)
        stride_map = list(target.stride_map)

        def fn(a):
            left = require_layout(a + 1, device_size, stride_map)
            right = require_layout(a + 2, device_size, stride_map)
            return left + right

        result = torch.compile(fn)(x)
        torch.testing.assert_close(result.cpu(), x.cpu() * 2 + 3, atol=0.2, rtol=0.1)

    @pytest.mark.parametrize(
        "fn",
        [
            lambda a: torch.sin(a),
            lambda a: a.sum(dim=-1),
        ],
    )
    def test_unsupported_producer_fails(self, fn):
        x = torch.randn(2, 128, dtype=torch.float16).to("spyre")

        with pytest.raises(Exception, match="no supported compiled producer"):
            torch.compile(lambda a: require_layout(fn(a), [2, 2, 64], [128, 64, 1]))(x)

    @pytest.mark.parametrize(
        "device_size,stride_map,error",
        [
            ([2, 2], [128], "equal nonzero lengths"),
            ([2, 0, 64], [128, 64, 1], "extents must be positive"),
            ([1, 2, 64], [128, 64, 1], "cannot hold tensor output"),
            ([2, 2, 64], [128, 64, -1], "must end in 1"),
            ([2, 2, 64], [128, 0, 1], "cannot broadcast"),
            ([2, 2, 64], [-2, 64, 1], "must be at least -1"),
            ([2, 2, 64], [64, 64, 1], "must be injective"),
        ],
    )
    def test_rejects_invalid_geometry(self, device_size, stride_map, error):
        x = torch.randn(2, 128, dtype=torch.float16).to("spyre")

        with pytest.raises(Exception, match=error):
            torch.compile(lambda a: require_layout(a + 1, device_size, stride_map))(x)
