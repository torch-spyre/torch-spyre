import torch
import torch_spyre._inductor.passes as passes

# import torch_spyre._inductor.propagate_real_dims as prd
import torch_spyre._inductor.propagate_real_dims as prd

passes.propagate_real_dims = prd.propagate_real_dims
declare_tensor_dim = prd.declare_tensor_dim
name_tensor_dims = prd.name_tensor_dims

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

declare_tensor_dim("A", d1)
declare_tensor_dim("B", d2)

name_tensor_dims(a_device, ["A", "B"])
name_tensor_dims(b_device, ["A", "B"])
name_tensor_dims(c_device, ["A", "B"])

compiled_result = torch.compile(func)(a_device, b_device, c_device).cpu()

# print(cpu_result)
# print(compiled_result)
# print(torch.abs(cpu_result - compiled_result).amax())
