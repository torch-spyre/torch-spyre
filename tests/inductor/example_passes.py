import torch

import torch._inductor.config as config
from torch_spyre._inductor.passes import CustomPostPasses

from torch._inductor.exc import InductorError

# to make this a subclass of InductorError, requires some more structure
class SpyreInductorEarlyExit(Exception):
    pass

def exit_fn(args):
    raise SpyreInductorEarlyExit("hello there")

def target_fn(a, b):
    return torch.matmul(a, b)

def test_pass(graph: torch.fx.Graph) -> None:
    print(list(graph.nodes))
    for node in list(graph.nodes):
        print("Op: ", node.op, " Target: ", node.target)
        if node.op == "call_function" and node.target == torch.matmul:
            print("Checking matmul args")

            arg = node.args[0]
            shape = arg.meta["example_value"].shape
            # take the second dimension
            assert shape[1] % 64 == 0, (
                "Expected 2nd dimension of arg0 to have been padded"
            )

config.force_disable_caches = True
torch.compiler.reset()
torch._dynamo.config.suppress_errors = False 

a = torch.empty((10, 2), device="spyre", dtype=torch.float16)
b = torch.empty((2, 10), device="spyre", dtype=torch.float16)

CustomPostPasses.passes.insert(1, test_pass)
CustomPostPasses.passes.insert(2, exit_fn)

# replace with an expect here
try:
    cmp = torch.compile(target_fn)
    res = cmp(a, b)
except InductorError as e:
    print(type(e.inner_exception))
    print("Finished test by skipping device operation")

# remove the exit fn
CustomPostPasses.passes.pop(2)
# remove the test fn
CustomPostPasses.passes.pop(1)
