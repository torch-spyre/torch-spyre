import torch

print("On CPU------->")
# some shapes to try
# 2d 
a = torch.randn((10, 10), device="cpu", dtype=torch.float16)
# 3d
a = torch.randn((2, 3, 5), device="cpu", dtype=torch.float16)

print(a)
print(f"A: {a.shape}, {a.stride()}")
b = a.sum(-1)
print(f"B: {b.shape}, {b.stride()}")
print("Result moved to Spyre: ")
c = b.to("spyre")
print(f"C: {c.shape}, {c.stride()}, {c.device_tensor_layout().device_size}")
print(c)

print("On Spyre----->")
a = a.to("spyre")
print(f"A: {a.shape}, {a.stride()}, {a.device_tensor_layout().device_size}")
print(a)
b = a.sum(-1)
print(f"B: {b.shape}, {b.stride()}, {b.device_tensor_layout().device_size}")
print(b)

@torch.compile
def fun(inp):
    return torch.ops.spyre.compact(inp)
c = fun(b)
print(f"C: {c.shape}, {c.stride()}, {c.device_tensor_layout().device_size}")
print(c)
