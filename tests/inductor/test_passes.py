import pytest
import unittest
import torch

from utils_inductor import (
    ParameterizedTestMeta,
    cached_randn,
)

from torch._dynamo.testing import (
    InductorAndRecordGraphs,
    normalize_gm,
)
import torch._inductor.config as config

def run_inject_test_pass(target_fn, pass_fn, pass_class, args):
    config.force_disable_caches = True
    torch.compiler.reset()
    backend = InductorAndRecordGraphs()

    pass_class.passes.append(pass_fn)
    cmp = torch.compile(target_fn, backend=backend)
    cmp(*args)
    pass_class.passes.pop()

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

        def test(x, y):
            return torch.matmul(x, y)

        def test_pass(graph: torch.fx.Graph) -> None:
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

        from torch_spyre._inductor.passes import CustomPreGradPasses

        run_inject_test_pass(test,
                             test_pass,
                             CustomPreGradPasses,
                             [x, y])
