import torch
import torch_spyre._inductor.passes as passes

# import torch_spyre._inductor.propagate_real_dims as prd
import torch_spyre._inductor.propagate_real_dims2 as prd

passes.propagate_real_dims = prd.propagate_real_dims
declare_real_dim = prd.declare_real_dim
annotate_real_dims = prd.annotate_real_dims

torch.manual_seed(0xAFFE)

x = torch.rand(64, 128, dtype=torch.float16)
y = torch.rand(128, 256, dtype=torch.float16)
w = torch.rand(256, 64, dtype=torch.float16)
z = torch.rand(64, 64, dtype=torch.float16)


def f(x, y, w, z):
    return x @ y @ w + z


r = f(x, y, w, z)
x_dev = x.to("spyre")
y_dev = y.to("spyre")
w_dev = w.to("spyre")
z_dev = z.to("spyre")

declare_real_dim("a", 64)
declare_real_dim("b", 128)
declare_real_dim("c", 256)
declare_real_dim("d", 64)

annotate_real_dims(x_dev, ["a", "b"])
annotate_real_dims(y_dev, ["b", "c"])
annotate_real_dims(w_dev, ["c", "d"])
annotate_real_dims(z_dev, ["a", "d"])

result = torch.compile(f)(x_dev, y_dev, w_dev, z_dev).cpu()

print(r)
print(result)
print(torch.abs(r - result).amax())
