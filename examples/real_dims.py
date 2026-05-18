import torch
from torch_spyre._inductor.propagate_real_dims import (
    declare_real_dim,
    annotate_real_dims,
)

torch.manual_seed(0xAFFE)

x = torch.rand(64 * 128, dtype=torch.float16).reshape(64, 128)
y = torch.rand(128, 256, dtype=torch.float16)
z = torch.rand(8, 8, 256, dtype=torch.float16).reshape(64, 256)


def f(x, y, z):
    return x @ y + z


r = f(x, y, z)
x_dev = x.to("spyre")
y_dev = y.to("spyre")
z_dev = z.to("spyre")

declare_real_dim("a", 64)
declare_real_dim("b", 128)
declare_real_dim("c", 256)

annotate_real_dims(x_dev, ["a", "b"])
annotate_real_dims(y_dev, ["b", "c"])
annotate_real_dims(z_dev, ["a", "c"])

z = torch.compile(f)(x_dev, y_dev, z_dev).cpu()

print(r)
print(z)
print(torch.abs(r - z).amax())
