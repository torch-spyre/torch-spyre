import torch
import torch_spyre._inductor.passes as passes

import torch_spyre._inductor.propagate_real_dims2 as prd

passes.propagate_real_dims = prd.propagate_real_dims

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

A, B = 3, 128
AB = A * B
a = torch.randn((AB,), dtype=torch.float16) * 0.1
b = torch.randn((AB,), dtype=torch.float16) * 0.1
c = torch.randn((A, B), dtype=torch.float16) * 0.1


def func(a, b, c):
    z = a + b
    return (z.view(A, B) + b.view(A, B)) + c


cpu_result = func(a, b, c)

a_device = a.to(DEVICE)
b_device = b.to(DEVICE)
c_device = c.to(DEVICE)

compiled_result = torch.compile(func)(a_device, b_device, c_device).cpu()

cpu_delta = torch.abs(compiled_result - cpu_result).max()
print(f"Max delta: {cpu_delta}")

try:
    torch.testing.assert_close(
        compiled_result,
        cpu_result,
        equal_nan=True,
        atol=0.1,
        rtol=0.1,
        msg=lambda msg: f"compiled spyre <-> cpu mismatch\n\n{msg}\n",
    )
    print("ANSWER CORRECT!")
except AssertionError as e:
    print(e)
