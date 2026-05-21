import torch
import torch_spyre._inductor.passes as passes

# import torch_spyre._inductor.propagate_real_dims as prd
import torch_spyre._inductor.propagate_real_dims as prd

passes.propagate_real_dims = prd.propagate_real_dims
declare_real_dim = prd.declare_real_dim
annotate_real_dims = prd.annotate_real_dims

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

declare_real_dim("A", A)
declare_real_dim("B", B)
declare_real_dim("C", C)
declare_real_dim("D", D)

annotate_real_dims(x_dev, ["A", "B"])
annotate_real_dims(y_dev, ["B", "C"])
annotate_real_dims(w_dev, ["C", "D"])
annotate_real_dims(z_dev, ["A", "D"])

result = torch.compile(f)(x_dev, y_dev, w_dev, z_dev).cpu()

print(r)
print(result)
print(torch.abs(r - result).amax())
