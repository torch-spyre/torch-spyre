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

    @pytest.mark.filterwarnings("ignore::torch_spyre.fallbacks.FallbackWarning")
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
            t = torch.sin(t)  # fallback op
            return t

        before_decomps = copy.deepcopy(decompositions)
        before_post_autograd_decomposition_table = copy.deepcopy(
            global_decomposition_table["post_autograd"]
        )
        before_pre_autograd_decomposition_table = copy.deepcopy(
            global_decomposition_table["pre_autograd"]
        )

        with pytest.warns(torch_spyre.fallbacks.FallbackWarning) as record:
            compare_with_cpu(fn, x, cpu_compile=True)

        assert len(record) == 1, "Exactly one FallbackWarning should be encountered!"

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

    @pytest.mark.filterwarnings("ignore::torch_spyre.fallbacks.FallbackWarning")
    @pytest.mark.filterwarnings(
        "ignore::UserWarning"
    )  # because of forced cache disabling
    def test_decompositions_graph(self):
        from torch._dynamo.testing import (
            InductorAndRecordGraphs,
            normalize_gm,
        )
        import torch._inductor.config as config

        # Disable all Inductor caches
        config.force_disable_caches = True

        def _check_out(out, tar):
            torch.testing.assert_close(
                out,
                tar,
                equal_nan=True,
                atol=0.1,
                rtol=0.1,
                msg=lambda msg: f"compiled spyre <-> compiled cpu mismatch\n\n{msg}\n",
            )

        def fn(device):
            t = torch.arange(65, device=device)
            return t

        # For the `cpu`, there is a decomposition for arange which should be captured
        torch.compiler.reset()
        backend = InductorAndRecordGraphs()
        cmp = torch.compile(fn, backend=backend)
        out = cmp("cpu")
        _check_out(out, torch.arange(65))
        expected_graph_str = """\
class <lambda>(torch.nn.Module):
    def forward(self):
        iota: "i64[65]" = torch.ops.prims.iota.default(65, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        return (iota,)
"""
        inductor_graph_str = normalize_gm(
            backend.inductor_graphs[0].print_readable(print_output=False)
        )
        assert inductor_graph_str == expected_graph_str, "Graphs are not identical"

        # For `spyre`, there is NO decomposition for arange
        torch.compiler.reset()
        backend = InductorAndRecordGraphs()
        cmp = torch.compile(fn, backend=backend)
        out = cmp("spyre")
        _check_out(out.cpu(), torch.arange(65))
        expected_graph_str = """\
class <lambda>(torch.nn.Module):
    def forward(self):
        arange: "i64[65]" = torch.ops.aten.arange.default(65, device = device(type='spyre'), pin_memory = False)
        return (arange,)
"""
        inductor_graph_str = normalize_gm(
            backend.inductor_graphs[0].print_readable(print_output=False)
        )
        assert inductor_graph_str == expected_graph_str, "Graphs are not identical"

        # Check that `cpu` still has the decomposition registered
        torch.compiler.reset()
        backend = InductorAndRecordGraphs()
        cmp = torch.compile(fn, fullgraph=True, backend=backend)
        out = cmp("cpu")
        _check_out(out, torch.arange(65))
        expected_graph_str = """\
class <lambda>(torch.nn.Module):
    def forward(self):
        iota: "i64[65]" = torch.ops.prims.iota.default(65, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        return (iota,)
"""
        inductor_graph_str = normalize_gm(
            backend.inductor_graphs[0].print_readable(print_output=False)
        )
        assert inductor_graph_str == expected_graph_str, "Graphs are not identical"

        def fn(device):
            t1 = torch.arange(65, device=device).to("cpu")  # noqa: F841
            t2 = torch.arange(65, device="cpu")
            return t2

        # In case the graph contains `spyre` and `cpu` tensors,
        # the decompositons of `spyre` are used and thus the
        # `cpu` arange is not used anymore.
        # Note: This is a known limitation at the moment as the
        # spyre-specific lowerings and decompositions are
        # merged PER GRAPH not per instruction
        torch.compiler.reset()
        backend = InductorAndRecordGraphs()
        cmp = torch.compile(fn, backend=backend)
        out = cmp("spyre")
        _check_out(out, torch.arange(65))
        expected_graph_str = """\
class <lambda>(torch.nn.Module):
    def forward(self):
        arange_1: "i64[65]" = torch.ops.aten.arange.default(65, device = device(type='cpu'), pin_memory = False)
        return (arange_1,)
"""
        inductor_graph_str = normalize_gm(
            backend.inductor_graphs[0].print_readable(print_output=False)
        )
        assert inductor_graph_str == expected_graph_str, "Graphs are not identical"

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


if __name__ == "__main__":
    unittest.main()
