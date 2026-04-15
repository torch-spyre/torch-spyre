import pytest
import unittest
import torch

from utils_inductor import (
    ParameterizedTestMeta,
    cached_randn,
)

class TestOps(unittest.TestCase, metaclass=ParameterizedTestMeta):
    torch.manual_seed(0xAFFE)

    PARAMS = {
        (
            "test_matmul_padding", "test_matmul_padding"
        ): {
            "param_sets": {
                "2d": (
                    cached_randn((10, 2), dtype=torch.float16).to("spyre"),
                    cached_randn((2, 20), dtype=torch.float16).to("spyre"),
                )
            }
        }
    }

    def test_matmul_padding(self, x: torch.Tensor, y: torch.Tensor):

        from torch._dynamo.testing import (
            InductorAndRecordGraphs,
            normalize_gm,
        )

        import torch._inductor.config as config
        config.force_disable_caches = True
        torch.compiler.reset()
        backend = InductorAndRecordGraphs()

        def test(x, y):
            return torch.matmul(x, y)

        def fn(graph: torch.fx.Graph) -> None:
            print("Hi from Custom pass")

            for node in list(graph.nodes):
                if node.op == "call_function" and node.target == torch.matmul:
                    print("Checking matmul args")

                    # there are some checks in the original, which we skip here
                    # TODO: those maybe needed, not sure what cases trigger those

                    # we have some hardcoding here for the given example
                    # TODO: generalize

                    # look at first argument
                    arg = node.args[0]
                    shape = arg.meta["example_value"].shape
                    # take the second dimension
                    assert shape[1] % 64 == 0, (
                        "Expected 2nd dimension of arg0 to have been padded"
                    )

                    # now, let's examine the second argument
                    arg = node.args[1]
                    shape = arg.meta["example_value"].shape
                    # take the first dimension
                    assert shape[0] % 64 == 0, (
                        "Expected 1st dimension of arg1 to have been padded"
                    )

        # this doesn't seem to work
        # with torch._inductor.config.patch(pre_grad_custom_pass=fn):
        #     cmp = torch.compile(test, backend=backend)
        #     cmp(x, y)

        from torch_spyre._inductor.passes import CustomPreGradPasses

        # a bit ugly, convert into context manager
        CustomPreGradPasses.passes.append(fn)
        # core logic that should always exist
        cmp = torch.compile(test, backend=backend)
        cmp(x, y)
        # convert into context manager
        CustomPreGradPasses.passes.pop()

        inductor_graph_str = normalize_gm(
            backend.inductor_graphs[0].print_readable(print_output=False)
        )

        # examine the full output if needed
        # print(inductor_graph_str)

        # one loose option is to check for properties here
        # !!! HARDCODED - CHANGE THIS !!!
        # but this doesn't work with the way we have parameterized this function
        assert "[10, 64]" in inductor_graph_str, (
            "Expected 2nd dimension of arg0 to have been padded"
        )

        assert "[64, 20]" in inductor_graph_str, (
            "Expected 1st dimension of arg1 to have been padded"
        )
