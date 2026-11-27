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
from utils_inductor import compare_with_cpu

FP32_EPS = torch.finfo(torch.float32).eps  # 1.1920928955078125e-07
FP16_EPS = torch.finfo(torch.float16).eps  # 0.0009765625


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
            "test_decompositions_change",
            "test_decompositions_change",
        ): {
            "param_sets": {
                "1d": (cached_randn((128,), dtype=torch.float16),),
            },
        },
        (
            "test_decompositions_graph",
            "test_decompositions_graph",
        ): {
            "param_sets": {
                "1d": (),
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_decompositions_change(self, x):
        import torch_spyre
        import types
        import copy
        from functools import partial
        from torch._inductor.decomposition import decompositions
        from torch._decomp import global_decomposition_table

        def _check_decomps(before, after):
            assert len(before.items()) == len(after.items()), (
                f"Amount of decompositions before ({len(before.items())}) and after ({len(after.items())}) not identical!"
            )

            for op, fn in before.items():
                if op._name not in [o._name for o in after.keys()]:
                    raise Exception(f"Decomposition {op} not present anymore!")
                else:
                    opa = None
                    for opa in after.keys():
                        if opa._name == op._name:
                            break
                    if isinstance(fn, types.FunctionType):
                        # fn is a regular function -> compare the hashes directly
                        if hash(fn) != hash(after[opa]):
                            raise Exception(
                                f"Decomposition for {op} changed!\nUsed to be {fn} and is now {after[opa]}"
                            )
                    elif isinstance(fn, partial):
                        # fn is a functools.partial -> compare the hashes of the functions
                        if hash(fn.func) != hash(after[opa].func):
                            raise Exception(
                                f"Decomposition for {op} changed!\nUsed to be {fn} and is now {after[opa]}"
                            )
                    else:
                        raise Exception(
                            f"Unexpected object in decomposition op: {op}, fn: {fn}"
                        )

        def fn(t):
            # aten.isin is a CPU fallback (torch_spyre/ops/fallbacks.py);
            # aten.cumsum no longer is -- its Spyre decomposition stays on
            # device, so it emits no FallbackWarning to count.
            return torch.isin(t, t[:1])  # fallback op

        before_decomps = copy.deepcopy(decompositions)
        before_post_autograd_decomposition_table = copy.deepcopy(
            global_decomposition_table["post_autograd"]
        )
        before_pre_autograd_decomposition_table = copy.deepcopy(
            global_decomposition_table["pre_autograd"]
        )

        with pytest.warns(torch_spyre.ops.fallbacks.FallbackWarning) as record:
            # run_eager=False: the eager path also triggers a FallbackWarning
            # (isin dispatches to the CPU fallback kernel), and pytest.warns
            # overrides the module-level "once" filter, so we'd capture 2
            # warnings instead of 1.  This test only cares about decomposition
            # table integrity, so skip eager.
            compare_with_cpu(fn, x, cpu_compile=True, run_eager=False)

        # pytest.warns records every warning raised in the block, not just
        # FallbackWarning (e.g. torch's once-per-process dynamo_pgo UserWarning
        # when caches are force-disabled), so count only the category we check.
        fallback_warnings = [
            w
            for w in record
            if issubclass(w.category, torch_spyre.ops.fallbacks.FallbackWarning)
        ]
        assert len(fallback_warnings) == 1, (
            "Exactly one FallbackWarning should be encountered!"
        )

        after_decomps = copy.deepcopy(decompositions)
        after_post_autograd_decomposition_table = copy.deepcopy(
            global_decomposition_table["post_autograd"]
        )
        after_pre_autograd_decomposition_table = copy.deepcopy(
            global_decomposition_table["pre_autograd"]
        )

        _check_decomps(before_decomps, after_decomps)
        _check_decomps(
            before_post_autograd_decomposition_table,
            after_post_autograd_decomposition_table,
        )
        _check_decomps(
            before_pre_autograd_decomposition_table,
            after_pre_autograd_decomposition_table,
        )

    @pytest.mark.filterwarnings(
        "ignore::UserWarning"
    )  # because of forced cache disabling
    def test_decompositions_graph(self):
        """Spyre's decompositions apply to a graph that uses Spyre, and only to it.

        ``randn`` is the probe: Inductor decomposes it into its random
        primitives on the host, while Spyre leaves it whole. The values are
        random, so only the captured graphs are compared.
        """
        from torch._dynamo.testing import (
            InductorAndRecordGraphs,
            normalize_gm,
        )
        import torch._inductor.config as config

        # Disable all Inductor caches
        config.force_disable_caches = True

        def graph_of(fn, device, **compile_kwargs):
            torch.compiler.reset()
            backend = InductorAndRecordGraphs()
            torch.compile(fn, backend=backend, **compile_kwargs)(device)
            return normalize_gm(
                backend.inductor_graphs[0].print_readable(print_output=False)
            )

        def fn(device):
            return torch.randn(65, device=device)

        # The host keeps Inductor's decomposition.
        cpu_graph = graph_of(fn, "cpu")
        assert "prims.inductor_random" in cpu_graph, cpu_graph
        assert "aten.randn" not in cpu_graph, cpu_graph

        # Spyre uses its own table, which does not decompose randn.
        spyre_graph = graph_of(fn, "spyre")
        assert "aten.randn.default" in spyre_graph, spyre_graph
        assert "prims.inductor_random" not in spyre_graph, spyre_graph

        # Compiling for Spyre leaves the host's table unchanged.
        assert graph_of(fn, "cpu", fullgraph=True) == cpu_graph

        def mixed(device):
            t1 = torch.randn(65, device=device).to("cpu")  # noqa: F841
            return torch.randn(65, device="cpu")

        # In a graph holding both, the table is chosen per graph rather than per
        # op, so the host randn is left whole too.
        # Note: This is a known limitation at the moment.
        mixed_graph = graph_of(mixed, "spyre")
        assert "torch.ops.aten.randn.default([65], device = device(type='cpu')" in (
            mixed_graph
        ), mixed_graph
        assert "prims.inductor_random" not in mixed_graph, mixed_graph

    @pytest.mark.filterwarnings(
        "ignore::UserWarning"
    )  # because of forced cache disabling
    def test_custom_decomposition_kwarg(self):
        """
        Test that custom decompositions passed as kwargs are properly handled for Spyre.
        This test creates a custom decomposition that decomposes add as multiplication,
        and verifies it's applied during compilation alongside Spyre-specific decompositions.
        """
        import torch._inductor.config as config
        from torch._dynamo.testing import normalize_gm

        # Disable all Inductor caches
        config.force_disable_caches = True

        # Create a custom decomposition table that decomposes add as mul
        custom_decomps = {}

        @torch._decomp.register_decomposition(
            [torch.ops.aten.add.Tensor], custom_decomps
        )
        def custom_add_decomp(x, y):
            # Decompose add as multiplication (for testing purposes)
            return torch.mul(x, y)

        # Custom backend that accepts decompositions
        class InductorAndRecordGraphsWithDecomps:
            def __init__(self, decompositions=None):
                self.graphs = []
                self.inductor_graphs = []
                self.decompositions = decompositions

            def __call__(self, gm, example_inputs):
                import torch._inductor.compile_fx as compile_fx_mod
                from unittest.mock import patch

                self.graphs.append(gm)

                old_compile_fx_inner = compile_fx_mod._compile_fx_inner

                def patched(*args, **kwargs):
                    self.inductor_graphs.append(args[0])
                    return old_compile_fx_inner(*args, **kwargs)

                with patch.object(compile_fx_mod, "_compile_fx_inner", new=patched):
                    return compile_fx_mod.compile_fx(
                        gm, example_inputs, decompositions=self.decompositions
                    )

        def fn(x, y):
            return torch.add(x, y)

        # Test with custom decomposition on Spyre
        torch.compiler.reset()
        backend = InductorAndRecordGraphsWithDecomps(decompositions=custom_decomps)
        cmp = torch.compile(fn, backend=backend)

        for device in ["cpu", "spyre"]:
            x = torch.randn(10).to(device=device)
            y = torch.randn(10).to(device=device)
            out = cmp(x, y)

            # Verify the graph uses mul instead of add
            inductor_graph_str = normalize_gm(
                backend.inductor_graphs[0].print_readable(print_output=False)
            )
            # The custom decomposition should have converted add to mul
            assert "mul" in inductor_graph_str, (
                "Custom decomposition not applied: expected 'mul' in graph"
            )
            assert "add" not in inductor_graph_str, (
                "Custom decomposition not applied: 'add' still present in graph"
            )

            # Verify the result matches the custom decomposition (mul, not add)
            expected = torch.mul(x, y)
            torch.testing.assert_close(out.cpu(), expected.cpu())

    def test_bool_amax_amin(self):
        """Test amax/amin on boolean tensors"""
        x = torch.randint(low=0, high=2, size=(256, 256)).to(torch.bool)

        # Test amax and amin
        compare_with_cpu(lambda a: torch.amax(a), x)
        compare_with_cpu(lambda a: torch.amin(a), x)


if __name__ == "__main__":
    unittest.main()
