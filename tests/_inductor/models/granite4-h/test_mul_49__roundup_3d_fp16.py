import torch
import torch.nn as nn
import random
import numpy as np


def get_op_callable():
    return torch.mul


class OpModule(nn.Module):
    def __init__(self):
        super().__init__()
        op = get_op_callable()
        self.op = op

    def forward(self, *args):
        return self.op(*args)


def SimpleIterate(device=None):
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = OpModule().to(device)
    backend = "inductor"
    net_compile = net if device == "cpu" else torch.compile(net, backend=backend)
    var0 = torch.rand(torch.Size([64]), dtype=torch.float16).to(device)
    var1 = torch.rand(torch.Size([1, 11, 64]), dtype=torch.float16).to(device)
    inputs = [var0, var1]
    output = net_compile(*inputs)
    if isinstance(output, tuple):
        output = output[0]
    return output.detach()


def test_mul_49__roundup_3d_fp16():
    out_cpu = SimpleIterate(device="cpu")
    out_sen = SimpleIterate(device="spyre").to("cpu")

    atol = 0.1
    rtol = 0.1
    same = torch.allclose(out_cpu, out_sen, atol=atol, rtol=rtol)
    print("Output match:", same)

    if not same:
        max_diff = (out_cpu - out_sen).abs().max().item()
        print("Max diff:", max_diff)

    assert same, f"Outputs are different: max diff={max_diff}"


if __name__ == "__main__":
    test_mul_49__roundup_3d_fp16()
