import torch
import torch_spyre._inductor.passes as passes

# import torch_spyre._inductor.propagate_real_dims as prd
import torch_spyre._inductor.propagate_real_dims2 as prd

passes.propagate_real_dims = prd.propagate_real_dims
declare_real_dim = prd.declare_real_dim
annotate_real_dims = prd.annotate_real_dims

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

declare_real_dim("W", AB)
declare_real_dim("A", A)
declare_real_dim("B", B)

annotate_real_dims(a_device, ["W"])
annotate_real_dims(b_device, ["W"])
annotate_real_dims(c_device, ["A", "B"])

compiled_result = torch.compile(func)(a_device, b_device, c_device).cpu()
