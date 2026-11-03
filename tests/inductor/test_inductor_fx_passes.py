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
from torch import fx
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    FixedLayout,
    InputBuffer,
    MutationLayoutSHOULDREMOVE,
    Pointwise,
    StorageBox,
    TensorBox,
)
from torch._inductor.virtualized import V

from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.pass_utils import (
    _repoint_mutation_targets,
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

        # Mock optimization_hint to raise TypeError for tmp0
        with patch("torch_spyre._inductor.pass_utils.V") as mock_v:
            mock_v.graph.sizevars.optimization_hint.side_effect = TypeError(
                "Cannot convert symbols to int"
            )

            # Should NOT raise, should return original index
            result = concretize_index(index, loop_vars)
            assert result == index, f"Expected {index}, got {result}"


class TestStickReshapePivot(unittest.TestCase):
    """pivot_unaligned_stick_reshapes: reshapes that cut the stick dim off a
    stick boundary (W=160 rows of 2.5 sticks, issue #1353) are rewritten into
    an on-device stick pivot instead of failing (split) or miscompiling
    (merge)."""

    def test_kind_classification(self):
        from torch_spyre._inductor.stick_reshape_pivot import _stick_reshape_kind

        kind = _stick_reshape_kind
        self.assertEqual(
            kind([1, 32, 25600], [1, 32, 160, 160], 64), ("split", [1, 32])
        )
        self.assertEqual(
            kind([1, 32, 160, 160], [1, 32, 25600], 64), ("merge", [1, 32])
        )
        self.assertEqual(
            kind([2, 16, 3200], [2, 16, 4, 20, 40], 64), ("split", [2, 16])
        )
        self.assertEqual(kind([6400], [80, 80], 64), ("split", []))
        # Outer dims reshaped too: handled as a separate plain view.
        self.assertEqual(kind([32, 25600], [1, 32, 160, 160], 64), ("split", [1, 32]))
        self.assertEqual(kind([2, 16, 25600], [32, 160, 160], 64), ("split", [32]))
        self.assertEqual(kind([32, 160, 160], [1, 32, 25600], 64), ("merge", [32]))
        # Sub-stick reshape (64 -> 8x8) is not free either: the destination
        # pads each row to a whole stick, so (8, 8) spans 8 sticks where (64,)
        # spans 1. Rewritten like any other unaligned reshape.
        self.assertEqual(kind([64], [8, 8], 64), ("split", []))
        self.assertEqual(kind([8, 8], [64], 64), ("merge", []))
        self.assertEqual(kind([4, 64], [4, 8, 8], 64), ("split", [4]))
        # Inside one stick but 7 does not divide 64 (#5198/#5214): rewritten.
        self.assertEqual(kind([1, 2, 4, 8, 7], [1, 8, 56], 64), ("merge", [1, 2, 4]))
        # 6-dim merge: the shape that exposed the #5804 coordinate bug.
        self.assertEqual(
            kind([1, 2, 4, 8, 4, 7], [1, 8, 224], 64), ("merge", [1, 2, 4])
        )
        # Stick-aligned innermost piece: an ordinary view, left alone.
        self.assertIsNone(kind([1, 32, 16384], [1, 32, 128, 128], 64))
        self.assertIsNone(kind([1, 32, 128, 128], [1, 32, 16384], 64))
        # Stick dim untouched (only outer dims reshaped).
        self.assertIsNone(kind([4, 8, 160], [32, 160], 64))
        # Stick dim regrouped, neither a split nor a merge of it.
        self.assertIsNone(kind([6, 160], [4, 240], 64))

    def _compile_and_check(self, fn, *args, on_device, expect):
        """Compile on Spyre, compare with fp32 CPU, and check which repair ran.

        ``expect`` is "pivot", "cpu" or None (reshape left alone).
        """
        from torch._dynamo.testing import InductorAndRecordGraphs, normalize_gm
        import torch._inductor.config as config
        from torch_spyre._inductor import config as spyre_config

        torch.compiler.reset()
        backend = InductorAndRecordGraphs()
        ref = fn(*[a.float() for a in args])
        # Patched rather than assigned, so a disabled cache cannot leak into
        # whichever test happens to run next.
        with (
            config.patch(force_disable_caches=True),
            spyre_config.patch(stick_reshape_on_device=on_device),
        ):
            out = torch.compile(fn, backend=backend)(*[a.to("spyre") for a in args])
        # fp16 SiLU/add on randn inputs; errors stay well under 5e-2.
        torch.testing.assert_close(out.cpu().float(), ref, atol=5e-2, rtol=1e-2)
        graph = normalize_gm(
            backend.inductor_graphs[0].print_readable(print_output=False)
        )
        self.assertEqual("spyre.rebase_copy" in graph, expect == "pivot")
        self.assertEqual("spyre.reshape_via_cpu" in graph, expect == "cpu")

    @pytest.mark.filterwarnings("ignore::UserWarning")  # forced cache disabling
    def test_split_unaligned_stick(self):
        for w in (160, 40, 20):
            for on_device in (True, False):
                with self.subTest(w=w, on_device=on_device):
                    a = torch.randn(1, 32, w * w, dtype=torch.float16)
                    self._compile_and_check(
                        lambda a, w=w: torch.nn.functional.silu(a * 2).reshape(
                            1, 32, w, w
                        )
                        + 1,
                        a,
                        on_device=on_device,
                        expect="pivot" if on_device else "cpu",
                    )

    @pytest.mark.filterwarnings("ignore::UserWarning")  # forced cache disabling
    def test_merge_unaligned_stick(self):
        # Used to compile and silently read every row after the first from the
        # wrong address.
        for w in (160, 40, 20):
            for on_device in (True, False):
                with self.subTest(w=w, on_device=on_device):
                    x = torch.randn(1, 32, w, w, dtype=torch.float16)
                    self._compile_and_check(
                        lambda x, w=w: torch.nn.functional.silu(x * 2).reshape(
                            1, 32, w * w
                        )
                        + 1,
                        x,
                        on_device=on_device,
                        expect="pivot" if on_device else "cpu",
                    )

    @pytest.mark.filterwarnings("ignore::UserWarning")  # forced cache disabling
    def test_deep_merge_six_dims(self):
        # (1,2,4,8,4,7) -> (1,8,224): deeper stride chain than the 5-dim case.
        # Compiles to wrong values under PR #5804's _wrap_and_project.
        x = torch.randn(1, 2, 4, 8, 4, 7, dtype=torch.float16)
        self._compile_and_check(
            lambda x: x.reshape(1, 8, 224) * 1.0, x, on_device=True, expect="pivot"
        )

    @pytest.mark.filterwarnings("ignore::UserWarning")  # forced cache disabling
    def test_sub_stick_reshape(self):
        # 64 -> 8x8 splits one stick into 8 rows, each padded back up to a
        # whole stick. Not free, and unsupported upstream.
        with self.subTest("no pivot dim -> host"):
            self._compile_and_check(
                lambda a: a.reshape(8, 8) * 1.0,
                torch.randn(64, dtype=torch.float16),
                on_device=True,
                expect="cpu",
            )
        with self.subTest("spare dim -> pivot"):
            self._compile_and_check(
                lambda a: a.reshape(4, 8, 8) * 1.0,
                torch.randn(4, 64, dtype=torch.float16),
                on_device=True,
                expect="pivot",
            )

    @pytest.mark.filterwarnings("ignore::UserWarning")  # forced cache disabling
    def test_multi_stick_pivot_alone_falls_back(self):
        # A pivot spanning more than one stick with no other non-unit dim to
        # sit in front of the body: the restickify back reads it wrongly, so
        # the host path is taken. P <= 64 stays on device.
        for P, expect in ((64, "pivot"), (100, "cpu"), (128, "cpu")):
            with self.subTest(P=P):
                self._compile_and_check(
                    lambda a, P=P: a.reshape(P, 21) * 1.0,
                    torch.randn(P, 3, 7, dtype=torch.float16),
                    on_device=True,
                    expect=expect,
                )
        # ... but a real outer dim in front makes the same pivot safe.
        self._compile_and_check(
            lambda a: a.reshape(3, 100, 21) * 1.0,
            torch.randn(3, 100, 3, 7, dtype=torch.float16),
            on_device=True,
            expect="pivot",
        )

    @pytest.mark.filterwarnings("ignore::UserWarning")  # forced cache disabling
    def test_stick_dim_cat_falls_back(self):
        # A pivoted reshape read by a cat along the stick dim trips the cat
        # lowering's restickify restore (the operand's stick dim is padded,
        # 56 of 64), so the host path is used. A cat on any other dim, and an
        # aligned reshape, stay on device.
        a = torch.randn(1, 2, 4, 8, 7, dtype=torch.float16)
        b = torch.randn(1, 2, 4, 8, 7, dtype=torch.float16)
        with self.subTest("cat on stick dim -> host"):
            self._compile_and_check(
                lambda a, b: torch.cat([a.reshape(1, 8, 56), b.reshape(1, 8, 56)], -1)
                * 1.0,
                a,
                b,
                on_device=True,
                expect="cpu",
            )
        with self.subTest("cat on another dim -> pivot"):
            self._compile_and_check(
                lambda a, b: torch.cat([a.reshape(1, 8, 56), b.reshape(1, 8, 56)], 1)
                * 1.0,
                a,
                b,
                on_device=True,
                expect="pivot",
            )

    @pytest.mark.filterwarnings("ignore::UserWarning")  # forced cache disabling
    def test_outer_dims_reshaped_too(self):
        # (2, 16, L) -> (32, H, W): the outer merge runs after the pivot; a view
        # in front of the first restickify fails in the backend.
        b = torch.randn(2, 16, 160 * 160, dtype=torch.float16)
        self._compile_and_check(
            lambda b: torch.nn.functional.silu(b * 2).reshape(32, 160, 160) + 1,
            b,
            on_device=True,
            expect="pivot",
        )

    @pytest.mark.filterwarnings("ignore::UserWarning")  # forced cache disabling
    def test_aligned_stick_reshape_is_left_alone(self):
        a = torch.randn(1, 32, 128 * 128, dtype=torch.float16)
        self._compile_and_check(
            lambda a: torch.nn.functional.silu(a * 2).reshape(1, 32, 128, 128) + 1,
            a,
            on_device=True,
            expect=None,
        )

    @pytest.mark.filterwarnings("ignore::UserWarning")  # forced cache disabling
    def test_escaping_alias_is_left_alone(self):
        """A reshape is not repaired when that would break a visible alias.

        Both ``y`` and its view escape and share storage, so mutating one
        must still be visible through the other. Every repair in this pass
        materialises the view, so the reshape has to be left as it is.
        """
        import torch._inductor.config as config

        # Either end of the alias may be reached through further views, so
        # cover the base itself, a sibling view of it, and a view of the
        # reshape's own result.
        cases = {
            "base and view": lambda y: (y, y.view(4, 8, 8)),
            "sibling view and view": lambda y: (y.t(), y.view(4, 8, 8)),
            "base and view of view": lambda y: (
                y,
                y.view(4, 8, 8).transpose(1, 2),
            ),
        }
        for name, tail in cases.items():
            with self.subTest(name):
                torch.compiler.reset()
                with config.patch(force_disable_caches=True):
                    first, second = torch.compile(
                        lambda x, tail=tail: tail(x * 2), dynamic=False
                    )(torch.ones(4, 64, dtype=torch.float16).to("spyre"))
                first.fill_(7)
                self.assertEqual(second.cpu().flatten()[:4].tolist(), [7.0] * 4)


class TestPassUtils(unittest.TestCase):
    """Unit tests for helpers in ``torch_spyre._inductor.pass_utils``."""

    @staticmethod
    def _mock_v(lower=None, upper=None, optimization_hint=None):
        """Build a mock ``V`` whose ShapeEnv reports the given bounds.

        ``optimization_hint`` is wired only when provided; tests that should
        never reach the hint fallback can omit it (any accidental call would
        raise ``AttributeError`` and fail the test loudly).
        """
        shape_env = SimpleNamespace(
            bound_sympy=lambda _e: SimpleNamespace(lower=lower, upper=upper)
        )
        sizevars = SimpleNamespace(shape_env=shape_env)
        if optimization_hint is not None:
            sizevars.optimization_hint = lambda _e: optimization_hint
        return SimpleNamespace(graph=SimpleNamespace(sizevars=sizevars))

    def test_compute_max_size(self):
        # Branches that never touch V.graph
        assert compute_max_size(42) == 42
        assert compute_max_size(sympy.Integer(7)) == 7
        assert compute_max_size(sympy.Integer(3) + sympy.Integer(4)) == 7

        s = sympy.Symbol("s0", integer=True, positive=True)

        # Finite ShapeEnv upper bound is recorded, return it.
        # ``optimization_hint`` is wired to a deliberately wrong value so a
        # regression that falls through to the hint would fail loudly.
        mock_v = self._mock_v(upper=sympy.Integer(576), optimization_hint=9999)
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            assert compute_max_size(s) == 576

        # No usable upper bound (sympy.oo) -- fall back to optimization_hint.
        mock_v = self._mock_v(upper=sympy.oo, optimization_hint=64)
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            assert compute_max_size(s) == 64

        # Edge case: the ``finite_upper_or_none`` predicate filters
        # non-positive bounds (``int(vr.upper) > 0``). Zero upper must
        # fall through to optimization_hint just like sympy.oo does.
        mock_v = self._mock_v(upper=sympy.Integer(0), optimization_hint=42)
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

    def test_compute_granularity_hint_fallback_emits_warning(self):
        # No finite upper bound -> max came from optimization_hint; helper warns.
        expr = sympy.Symbol("s0", integer=True, positive=True)
        mock_v = self._mock_v(sympy.Integer(2), sympy.oo)
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                granularity = compute_granularity(expr, max_size=1024)
        assert granularity == 32
        assert any("came from optimization_hint" in str(x.message) for x in w)

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
            optimization_hint=512,
        )
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            result = compute_symbolic_bounds(s0)
        assert result == (1024, 64)

    def test_compute_symbolic_bounds_infinite_upper_falls_back_to_hint(self):
        # dynamic=True without mark_dynamic(max=...) gives upper=oo.
        # max_size must fall back to optimization_hint, not oo. lower=2 is
        # ShapeEnv's default (no mark_dynamic(min=...) given), so
        # granularity comes from compute_granularity's default-divisor
        # branch: smallest divisor of 256 >= min_default_granularity(4)
        # with 256/d <= max_buckets(32) is 8.
        s0 = sympy.Symbol("s0", integer=True, positive=True)
        mock_v = self._mock_v(
            lower=sympy.Integer(2), upper=sympy.oo, optimization_hint=256
        )
        with patch("torch_spyre._inductor.pass_utils.V", mock_v):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = compute_symbolic_bounds(s0)
        assert result is not None
        max_size, granularity = result
        assert max_size == 256
        assert granularity == 8


class TestRepointMutationTargets(unittest.TestCase):
    """Unit tests for ``pass_utils._repoint_mutation_targets``.

    Regression coverage for issue #3944/#3945: reconstructing a
    ``ComputedBuffer`` (``replace_computed_buffer_body``,
    ``redirect_computed_buffer_reads``) must not leave a mutation op's
    ``MutationLayoutSHOULDREMOVE.target`` pointing at the orphaned
    pre-reconstruction object. These tests exercise ``_repoint_mutation_targets``
    directly against real IR objects, independent of any particular pass
    that calls it, so the fix stays pinned even if coarse-tiling (the pass
    that originally exposed the bug) is refactored away.
    """

    def setUp(self):
        gm = fx.symbolic_trace(lambda: None)
        self._graph_ctx = V.set_graph_handler(GraphLowering(gm))
        self._graph_ctx.__enter__()
        self.addCleanup(self._graph_ctx.__exit__, None, None, None)

    @staticmethod
    def _make_buffer(name):
        """A minimal real ComputedBuffer(Pointwise) reading one InputBuffer."""
        inp = InputBuffer(
            name=f"in_{name}",
            layout=FixedLayout(torch.device("cpu"), torch.float32, [8], [1]),
        )
        V.graph.name_to_buffer[inp.get_name()] = inp
        box = TensorBox(StorageBox(inp))
        pw = Pointwise.create(
            device=torch.device("cpu"),
            dtype=torch.float32,
            inner_fn=lambda index: box.make_loader()(index),
            ranges=[8],
        )
        buf = ComputedBuffer(
            name=name,
            layout=FixedLayout(torch.device("cpu"), torch.float32, [8], None),
            data=pw.data.data,  # TensorBox -> StorageBox -> Pointwise
        )
        buf.operation_name = name
        V.graph.name_to_buffer[name] = buf
        return buf

    def test_bare_target_repointed(self):
        old_buf = self._make_buffer("old")
        new_buf = self._make_buffer("new")
        mutation_layout = MutationLayoutSHOULDREMOVE(old_buf)
        mut_op = SimpleNamespace(layout=mutation_layout)

        _repoint_mutation_targets([mut_op], old_buf, new_buf)

        self.assertIs(mutation_layout.target, new_buf)

    def test_boxed_target_repointed(self):
        """target wrapped as TensorBox(StorageBox(old_buf))."""
        old_buf = self._make_buffer("old")
        new_buf = self._make_buffer("new")
        wrapped_target = TensorBox(StorageBox(old_buf))
        mutation_layout = MutationLayoutSHOULDREMOVE(wrapped_target)
        mut_op = SimpleNamespace(layout=mutation_layout)

        _repoint_mutation_targets([mut_op], old_buf, new_buf)

        # The wrapper object identity itself is untouched -- only the
        # innermost slot holding old_buf is repointed.
        self.assertIs(mutation_layout.target, wrapped_target)
        self.assertIs(wrapped_target.data.data, new_buf)

    def test_unrelated_op_untouched(self):
        """An op whose target already points elsewhere must not be touched."""
        old_buf = self._make_buffer("old")
        new_buf = self._make_buffer("new")
        other_buf = self._make_buffer("other")
        mutation_layout = MutationLayoutSHOULDREMOVE(other_buf)
        mut_op = SimpleNamespace(layout=mutation_layout)

        _repoint_mutation_targets([mut_op], old_buf, new_buf)

        self.assertIs(mutation_layout.target, other_buf)

    def test_non_mutation_op_untouched(self):
        """An op with a plain (non-mutation) layout must be skipped safely."""
        old_buf = self._make_buffer("old")
        new_buf = self._make_buffer("new")
        plain_op = SimpleNamespace(layout=old_buf.layout)

        # Must not raise even though `plain_op.layout` isn't a
        # MutationLayoutSHOULDREMOVE.
        _repoint_mutation_targets([plain_op], old_buf, new_buf)

        self.assertIs(plain_op.layout, old_buf.layout)


if __name__ == "__main__":
    unittest.main()
