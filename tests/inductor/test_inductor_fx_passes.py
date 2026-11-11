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

from torch_spyre._inductor.pass_utils import _repoint_mutation_targets
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
