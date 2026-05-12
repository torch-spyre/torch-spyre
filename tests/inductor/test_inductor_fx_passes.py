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
import torch

from utils_inductor import (
    ParameterizedTestMeta,
    cached_randn,
)

from torch_spyre._inductor.padding import insert_padding


def _make_mm_graph(x_shape, w_shape, dtype=torch.float16):
    """Build a minimal post-grad FX graph containing a single aten.mm.default node."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(x_shape, dtype=dtype, device="spyre")
    w = graph.placeholder("w")
    w.meta["val"] = torch.empty(w_shape, dtype=dtype, device="spyre")
    mm = graph.call_function(torch.ops.aten.mm.default, args=(x, w))
    mm.meta["val"] = torch.empty((x_shape[0], w_shape[1]), dtype=dtype, device="spyre")
    graph.output(mm)
    return graph


def _make_bmm_graph(x_shape, w_shape, dtype=torch.float16):
    """Build a minimal post-grad FX graph containing a single aten.bmm.default node."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(x_shape, dtype=dtype, device="spyre")
    w = graph.placeholder("w")
    w.meta["val"] = torch.empty(w_shape, dtype=dtype, device="spyre")
    bmm = graph.call_function(torch.ops.aten.bmm.default, args=(x, w))
    bmm.meta["val"] = torch.empty(
        (x_shape[0], x_shape[1], w_shape[2]), dtype=dtype, device="spyre"
    )
    graph.output(bmm)
    return graph


def _overwrite_nodes(graph):
    return [n for n in graph.nodes if n.target == torch.ops.spyre.overwrite_f.default]


class TestOps(unittest.TestCase, metaclass=ParameterizedTestMeta):
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


class TestBmmUnflattenWithResidual(unittest.TestCase):
    """Regression test for bmm unflatten pass with SDPA + Dense + LayerNorm.

    The QFormer projector pattern (SDPA followed by Dense + LayerNorm + residual)
    previously crashed with "batch1 must be a 3D tensor" in FakeTensorUpdater
    because bmm_unflatten_pass created aten.bmm nodes with 4D inputs.
    """

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_sdpa_dense_layernorm_compiles(self):
        """SDPA + Dense + LayerNorm + residual should compile without error."""
        import torch._inductor.config as config
        from torch._dynamo.testing import InductorAndRecordGraphs, normalize_gm

        config.force_disable_caches = True

        B, S, H, E = 1, 32, 12, 64
        hidden = H * E

        class QFormerAttentionLike(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.query = torch.nn.Linear(hidden, hidden, bias=False)
                self.key = torch.nn.Linear(hidden, hidden, bias=False)
                self.value = torch.nn.Linear(hidden, hidden, bias=False)
                self.dense = torch.nn.Linear(hidden, hidden, bias=False)
                self.layernorm = torch.nn.LayerNorm(hidden)

            def forward(self, x):
                residual = x
                q = self.query(x).view(B, S, H, E).transpose(1, 2)
                k = self.key(x).view(B, S, H, E).transpose(1, 2)
                v = self.value(x).view(B, S, H, E).transpose(1, 2)
                attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                attn = attn.transpose(1, 2).contiguous().view(B, S, hidden)
                out = self.dense(attn)
                out = self.layernorm(out + residual)
                return out

        model = QFormerAttentionLike().half().to("spyre").eval()
        x = torch.randn(B, S, hidden, dtype=torch.float16, device="spyre")

        torch.compiler.reset()
        backend = InductorAndRecordGraphs()
        compiled = torch.compile(model, backend=backend)
        compiled(x)

        graph_str = normalize_gm(
            backend.inductor_graphs[0].print_readable(print_output=False)
        )
        assert "spyre.batched_matmul" in graph_str, (
            "Expected spyre.batched_matmul for 4D SDPA matmuls, "
            f"but got graph:\n{graph_str}"
        )


class TestInsertPadding(unittest.TestCase):
    """Unit tests for the insert_padding post-grad FX pass."""

    def test_mm_unaligned_reduction_dim_gets_padded(self):
        # 200 is not a multiple of 64 (fp16 stick size), so both args need padding
        graph = _make_mm_graph(x_shape=(67, 200), w_shape=(200, 128))
        insert_padding(graph)
        overwrites = _overwrite_nodes(graph)
        self.assertEqual(len(overwrites), 2, "expected padding on both mm args")

    def test_mm_aligned_reduction_dim_no_padding(self):
        # 256 is a multiple of 64 — no padding needed on the reduction dim
        graph = _make_mm_graph(x_shape=(67, 256), w_shape=(256, 128))
        insert_padding(graph)
        overwrites = _overwrite_nodes(graph)
        self.assertEqual(
            len(overwrites), 0, "expected no padding for aligned reduction dim"
        )

    def test_bmm_unaligned_reduction_dim_gets_padded(self):
        # 200 is not a multiple of 64
        graph = _make_bmm_graph(x_shape=(2, 67, 200), w_shape=(2, 200, 128))
        insert_padding(graph)
        overwrites = _overwrite_nodes(graph)
        self.assertEqual(len(overwrites), 2, "expected padding on both bmm args")

    def test_mm_reduction_dim_1_skipped(self):
        # reduction dim == 1 is special-cased: lowering converts size-1 mm to mul
        graph = _make_mm_graph(x_shape=(67, 1), w_shape=(1, 128))
        insert_padding(graph)
        overwrites = _overwrite_nodes(graph)
        self.assertEqual(
            len(overwrites), 0, "size-1 reduction dim should not be padded"
        )

    def test_mm_padded_arg_has_correct_shape(self):
        # x is (67, 200): pad 200 → 256 along dim -1; w is (200, 128): pad 200 → 256 along dim -2
        graph = _make_mm_graph(x_shape=(67, 200), w_shape=(200, 128))
        insert_padding(graph)
        overwrites = _overwrite_nodes(graph)
        shapes = sorted([tuple(n.meta["val"].shape) for n in overwrites])
        self.assertIn((67, 256), shapes)
        self.assertIn((256, 128), shapes)


if __name__ == "__main__":
    unittest.main()
