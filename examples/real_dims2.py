import torch
import torch_spyre._inductor.passes as passes

# import torch_spyre._inductor.propagate_real_dims as prd
import torch_spyre._inductor.propagate_real_dims as prd

passes.propagate_real_dims = prd.propagate_real_dims
declare_tensor_dim = prd.declare_tensor_dim
name_tensor_dims = prd.name_tensor_dims

torch.manual_seed(0xAFFE)

A, B, C, D = 64, 128, 256, 64
x = torch.rand(A, B, dtype=torch.float16) * 0.01
y = torch.rand(B, C, dtype=torch.float16) * 0.01
w = torch.rand(C, D, dtype=torch.float16) * 0.01
z = torch.rand(A, D, dtype=torch.float16) * 0.01


def f(x, y, w, z):
    return (x @ y @ w + z).sum(0)


r = f(x, y, w, z)
x_dev = x.to("spyre")
y_dev = y.to("spyre")
w_dev = w.to("spyre")
z_dev = z.to("spyre")

declare_tensor_dim("A", A)
declare_tensor_dim("B", B)
declare_tensor_dim("C", C)
declare_tensor_dim("D", D)

name_tensor_dims(x_dev, ["A", "B"])
name_tensor_dims(y_dev, ["B", "C"])
name_tensor_dims(w_dev, ["C", "D"])
name_tensor_dims(z_dev, ["A", "D"])

result = torch.compile(f)(x_dev, y_dev, w_dev, z_dev).cpu()

print(r)
print(result)
print(torch.abs(r - result).amax())
