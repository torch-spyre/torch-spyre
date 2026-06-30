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

import pytest
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import sympy
import torch

from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.pass_utils import (
    compute_granularity,
    compute_max_size,
    compute_symbolic_bounds,
)
from utils_inductor import (
    ParameterizedTestMeta,
    cached_randn,
)


class TestOps(unittest.TestCase, metaclass=ParameterizedTestMeta):
    torch.manual_seed(0xAFFE)  # seeds cached_randn/cached_xavier calls in PARAMS below

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    # Define parameter sets for each base test method
    # If parameterized, the base test method will not be invoked
    # The test methods that are not parameterized will be invoked
    # as usual (i.e. no change in their behaviors)
    # If using unittest.skip decorator on a base function that is
    # parameterized, the parameterized functions are skipped too
    # See utils.py for more details.
    PARAMS = {
        (
            "test_linear_decomposition_graph",
            "test_linear_decomposition_graph",
        ): {
            "param_sets": {
                "2d": (
                    cached_randn((67, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((128, 256), dtype=torch.float16).to("spyre"),
                    None,
                ),
                "3d": (
                    cached_randn((2, 67, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((128, 256), dtype=torch.float16).to("spyre"),
                    None,
                ),
                "2d_bias": (
                    cached_randn((67, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((128, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((128,), dtype=torch.float16).to("spyre"),
                ),
                "3d_bias": (
                    cached_randn((67, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((128, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((128,), dtype=torch.float16).to("spyre"),
                ),
            },
        },
        (
            "test_unflatten_bmm_pass_graph",
            "test_unflatten_bmm_pass_graph",
        ): {
            "param_sets": {
                "3d_2d": (
                    cached_randn((2, 67, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((256, 128), dtype=torch.float16).to("spyre"),
                ),
                "3d_3d": (
                    cached_randn((2, 67, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((2, 256, 128), dtype=torch.float16).to("spyre"),
                ),
                "3d_3d_bcast": (
                    cached_randn((4, 67, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((1, 256, 128), dtype=torch.float16).to("spyre"),
                ),
                "4d_4d": (
                    cached_randn((3, 17, 128, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((3, 17, 256, 128), dtype=torch.float16).to("spyre"),
                ),
                "4d_4d_bcast": (
                    cached_randn((3, 1, 128, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((1, 17, 256, 128), dtype=torch.float16).to("spyre"),
                ),
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings(
        "ignore::UserWarning"
    )  # because of forced cache disabling
    def test_linear_decomposition_graph(
        self, x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None
    ):
        from torch._dynamo.testing import (
            InductorAndRecordGraphs,
            normalize_gm,
        )
        import torch._inductor.config as config

        config.force_disable_caches = True

        # 2D input: F.linear should decompose via transpose + mm (no addmm)
        def linear_test(x, w, bias=None):
            return torch.nn.functional.linear(x, w, bias)

        torch.compiler.reset()
        backend = InductorAndRecordGraphs()
        cmp = torch.compile(linear_test, backend=backend)
        cmp(x, w, bias)

        inductor_graph_str = normalize_gm(
            backend.inductor_graphs[0].print_readable(print_output=False)
        )

        if x.dim() == 2:
            assert "aten.mm.default" in inductor_graph_str, (
                "Expected aten.mm.default in 2D linear decomposition graph"
            )
        elif x.dim() == 3:
            assert "aten.bmm.default" in inductor_graph_str, (
                "Expected aten.bmm.default in 3D linear decomposition graph"
            )
        assert "aten.addmm" not in inductor_graph_str, (
            "Custom linear decomp should avoid addmm"
        )

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings(
        "ignore::UserWarning"
    )  # because of forced cache disabling
    def test_unflatten_bmm_pass_graph(self, x: torch.Tensor, w: torch.Tensor):
        from torch._dynamo.testing import (
            InductorAndRecordGraphs,
            normalize_gm,
        )
        import torch._inductor.config as config

        config.force_disable_caches = True

        # matmul: view→mm→view should be converted to bmm by _unflatten_mm_to_bmm
        def fn(x, w):
            return x @ w

        torch.compiler.reset()
        backend = InductorAndRecordGraphs()
        cmp = torch.compile(fn, backend=backend)
        cmp(x.to("spyre"), w.to("spyre"))

        inductor_graph_str = normalize_gm(
            backend.inductor_graphs[0].print_readable(print_output=False)
        )
        has_batched_matmul = (
            "aten.bmm.default" in inductor_graph_str
            or "spyre.batched_matmul" in inductor_graph_str
        )
        assert has_batched_matmul, (
            "Expected aten.bmm.default or spyre.batched_matmul after passes"
        )
        assert "aten.mm.default" not in inductor_graph_str, (
            "aten.mm.default should be replaced by bmm/batched_matmul after passes"
        )

    def test_mixed_device_seq(self):
        model = torch.compile(torch.sin)
        cpu_1 = torch._inductor.utils.get_code(model, torch.randn(5))[0]

        model = torch.compile(torch.sin)
        spyre_1 = torch._inductor.utils.get_code(model, torch.randn(5, device="spyre"))[
            0
        ]

        torch._dynamo.reset()
        model = torch.compile(torch.sin)
        cpu_2 = torch._inductor.utils.get_code(model, torch.randn(5))[0]

        assert cpu_1.split("\n", 1)[1] == cpu_2.split("\n", 1)[1], (
            "CPU graph should be the same across compilations"
        )
        assert spyre_1 != cpu_1, "SPYRE graph should differ from CPU graph"

    def test_concretize_index_with_symbolic_shapes(self):
        """
        Test that concretize_index handles unconvertible symbolic expressions.

        Regression test for: "TypeError: Cannot convert symbols to int"
        that occurred in index_copy operations with symbolic shapes.
        """
        from torch_spyre._inductor.pass_utils import concretize_index

        # Create symbolic variables
        x = sympy.Symbol("x")  # Loop variable
        tmp0 = sympy.Symbol("tmp0")  # Unconvertible symbol

        # Create expression: x + tmp0
        index = x + tmp0
        loop_vars = {x}

        # Mock size_hint to raise TypeError for tmp0
        with patch("torch_spyre._inductor.pass_utils.V") as mock_v:
            mock_v.graph.sizevars.size_hint.side_effect = TypeError(
                "Cannot convert symbols to int"
            )

            # Should NOT raise, should return original index
            result = concretize_index(index, loop_vars)
            assert result == index, f"Expected {index}, got {result}"


class TestPassUtils(unittest.TestCase):
    """Unit tests for helpers in ``torch_spyre._inductor.pass_utils``."""

    @staticmethod
    def _mock_v(lower=None, upper=None, size_hint=None):
        """Build a mock ``V`` whose ShapeEnv reports the given bounds.

        ``size_hint`` is wired only when provided; tests that should
        never reach the size_hint fallback can omit it (any accidental
        call would raise ``AttributeError`` and fail the test loudly).
        """
        shape_env = SimpleNamespace(
            bound_sympy=lambda _e: SimpleNamespace(lower=lower, upper=upper)
        )
        sizevars = SimpleNamespace(shape_env=shape_env)
        if size_hint is not None:
            sizevars.size_hint = lambda _e: size_hint
        return SimpleNamespace(graph=SimpleNamespace(sizevars=sizevars))

    def test_compute_max_size(self):
        # Branches that never touch V.graph
        assert compute_max_size(42) == 42
        assert compute_max_size(sympy.Integer(7)) == 7
        assert compute_max_size(sympy.Integer(3) + sympy.Integer(4)) == 7

        s = sympy.Symbol("s0", integer=True, positive=True)

        # Finite ShapeEnv upper bound is recorded, return it.
        # ``size_hint`` is wired to a deliberately wrong value so a
        # regression that falls through to size_hint would fail loudly.
        mock_v = self._mock_v(upper=sympy.Integer(576), size_hint=9999)
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            assert compute_max_size(s) == 576

        # No usable upper bound (sympy.oo) -- fall back to size_hint.
        mock_v = self._mock_v(upper=sympy.oo, size_hint=64)
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            assert compute_max_size(s) == 64

        # Edge case: the ``finite_upper_or_none`` predicate filters
        # non-positive bounds (``int(vr.upper) > 0``). Zero upper must
        # fall through to size_hint just like sympy.oo does.
        mock_v = self._mock_v(upper=sympy.Integer(0), size_hint=42)
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            assert compute_max_size(s) == 42

    def test_compute_granularity_user_min_happy_path(self):
        expr = sympy.Symbol("s0", integer=True, positive=True)
        mock_v = self._mock_v(sympy.Integer(16), sympy.Integer(512))
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            assert compute_granularity(expr, max_size=512) == 16

    def test_compute_granularity_user_min_not_a_divisor_raises(self):
        expr = sympy.Symbol("s0", integer=True, positive=True)
        mock_v = self._mock_v(sympy.Integer(7), sympy.Integer(512))
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            with self.assertRaises(Unsupported) as cm:
                compute_granularity(expr, max_size=512)
        assert "must divide max" in str(cm.exception)

    def test_compute_granularity_user_min_exceeds_bucket_cap_raises(self):
        expr = sympy.Symbol("s0", integer=True, positive=True)
        # min=4, max=512 -> 128 buckets > default cap of 32.
        mock_v = self._mock_v(sympy.Integer(4), sympy.Integer(512))
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            with self.assertRaises(Unsupported) as cm:
                compute_granularity(expr, max_size=512)
        msg = str(cm.exception)
        assert "buckets" in msg and "max_buckets" in msg

    def test_compute_granularity_default_with_warning(self):
        # lower=2 (PyTorch default) -> "no min"; smallest divisor of 1024
        # >= 4 with 1024/d <= 32 is 32.
        expr = sympy.Symbol("s0", integer=True, positive=True)
        mock_v = self._mock_v(sympy.Integer(2), sympy.Integer(1024))
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                granularity = compute_granularity(expr, max_size=1024)
        assert granularity == 32
        assert any("defaulting granularity" in str(x.message) for x in w)

    def test_compute_granularity_size_hint_fallback_emits_warning(self):
        # No finite upper bound -> max came from size_hint; helper warns.
        expr = sympy.Symbol("s0", integer=True, positive=True)
        mock_v = self._mock_v(sympy.Integer(2), sympy.oo)
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                granularity = compute_granularity(expr, max_size=1024)
        assert granularity == 32
        assert any("came from size_hint" in str(x.message) for x in w)

    def test_compute_symbolic_bounds_concrete_int_returns_none(self):
        with patch("torch_spyre._inductor.pass_utils.V", self._mock_v()):
            assert compute_symbolic_bounds(128) is None

    def test_compute_symbolic_bounds_concrete_sympy_integer_returns_none(self):
        with patch("torch_spyre._inductor.pass_utils.V", self._mock_v()):
            assert compute_symbolic_bounds(sympy.Integer(512)) is None

    def test_compute_symbolic_bounds_shape_env_none_returns_none(self):
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        sizevars = SimpleNamespace(shape_env=None)
        mock_v = SimpleNamespace(graph=SimpleNamespace(sizevars=sizevars))
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            assert compute_symbolic_bounds(s0) is None

    def test_compute_symbolic_bounds_finite_bounds(self):
        # lower=64 is a valid mark_dynamic(min=...): it divides max=1024
        # and stays within max_buckets, so compute_granularity honours it
        # as-is and the result coincides with the raw ShapeEnv lower bound.
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        mock_v = self._mock_v(
            lower=sympy.Integer(64),
            upper=sympy.Integer(1024),
            size_hint=512,
        )
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            result = compute_symbolic_bounds(s0)
        assert result == (1024, 64)

    def test_compute_symbolic_bounds_infinite_upper_falls_back_to_size_hint(self):
        # dynamic=True without mark_dynamic(max=...) gives upper=oo.
        # max_size must fall back to size_hint, not oo. lower=2 is
        # ShapeEnv's default (no mark_dynamic(min=...) given), so
        # granularity comes from compute_granularity's default-divisor
        # branch: smallest divisor of 256 >= min_default_granularity(4)
        # with 256/d <= max_buckets(32) is 8.
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        mock_v = self._mock_v(lower=sympy.Integer(2), upper=sympy.oo, size_hint=256)
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = compute_symbolic_bounds(s0)
        assert result is not None
        max_size, granularity = result
        assert max_size == 256
        assert granularity == 8


if __name__ == "__main__":
    unittest.main()
