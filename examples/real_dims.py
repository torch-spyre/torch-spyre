import torch
from torch_spyre._inductor.propagate_real_dims import annotate_real_dims

torch.manual_seed(0xAFFE)

x = torch.rand(64, 128, dtype=torch.float16)
y = torch.rand(128, 256, dtype=torch.float16)
z = torch.rand(64, 256, dtype=torch.float16)


def f(x, y, z):
    return x @ y + z


r = f(x, y, z)
x_dev = x.to("spyre")
y_dev = y.to("spyre")
z_dev = z.to("spyre")

annotate_real_dims(x_dev, [("a", 64), ("b", 128)])
annotate_real_dims(y_dev, [("b", 128), ("c", 256)])
annotate_real_dims(z_dev, [("a", 64), ("c", 256)])

z = torch.compile(f)(x_dev, y_dev, z_dev).cpu()

print(r)
print(z)
print(torch.abs(r - z).amax())
