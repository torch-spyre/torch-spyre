import torch

device = torch.device("spyre:0")
x = torch.zeros(8, 32, 32, dtype=torch.float16)
slots = torch.arange(4, dtype=torch.int32)
values = torch.ones(4, 1024, dtype=torch.float16)

def fn(x, slots, values):
    x = x.view(8, 1024)
    x.index_put_((slots,), values, accumulate=False)
    return x

torch.compile(fn)(x.to(device), slots.to(device), values.to(device))
# Unexpected stick expression Mod(d1, 32)
