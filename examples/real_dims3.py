import torch
import torch_spyre._inductor.passes as passes

# import torch_spyre._inductor.propagate_real_dims as prd
import torch_spyre._inductor.propagate_real_dims as prd

passes.propagate_real_dims = prd.propagate_real_dims
declare_real_dim = prd.declare_real_dim
annotate_real_dims = prd.annotate_real_dims

DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

d1, d2 = 3, 128
a = torch.randn((d1 * d2), dtype=torch.float16) * 0.1
b = torch.randn((d1 * d2), dtype=torch.float16) * 0.1
c = torch.randn((d1, d2), dtype=torch.float16) * 0.1


def func(a, b, c):
    z = a + b
    return (z.view(d1, d2) + b.view(d1, d2)) + c


cpu_result = func(a, b, c)

a_device = a.to(DEVICE)
b_device = b.to(DEVICE)
c_device = c.to(DEVICE)

declare_real_dim("A", d1)
declare_real_dim("B", d2)

annotate_real_dims(a_device, ["A", "B"])
annotate_real_dims(b_device, ["A", "B"])
annotate_real_dims(c_device, ["A", "B"])

compiled_result = torch.compile(func)(a_device, b_device, c_device).cpu()

# print(cpu_result)
# print(compiled_result)
# print(torch.abs(cpu_result - compiled_result).amax())
