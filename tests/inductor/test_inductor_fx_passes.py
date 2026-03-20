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
                # TODO(aviros): Fails on codegen
                # "3d_3d_bcast": (
                #     cached_randn((4, 67, 256), dtype=torch.float16).to("spyre"),
                #     cached_randn((1, 256, 128), dtype=torch.float16).to("spyre"),
                # ),
                "4d_4d": (
                    cached_randn((3, 17, 128, 256), dtype=torch.float16).to("spyre"),
                    cached_randn((3, 17, 256, 128), dtype=torch.float16).to("spyre"),
                ),
                # TODO(aviros): Fails on codegen
                # "4d_4d_bcast": (
                #     cached_randn((3, 1, 128, 256), dtype=torch.float16).to("spyre"),
                #     cached_randn((1, 17, 256, 128), dtype=torch.float16).to("spyre"),
                # ),
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
        assert "aten.bmm.default" in inductor_graph_str, (
            "Expected aten.bmm.default after unflatten_mm_to_bmm pass"
        )
        assert "aten.mm.default" not in inductor_graph_str, (
            "aten.mm.default should be replaced by bmm after unflatten pass"
        )


if __name__ == "__main__":
    unittest.main()
