import torch


DEVICE = torch.device("spyre")
torch.manual_seed(0xAFFE)

x = torch.rand(128, 64, dtype=torch.float16)
y = torch.rand(128, 64, dtype=torch.float16)

cpu_result = torch.mul(x, y)

x_device = x.to(DEVICE)
y_device = y.to(DEVICE)

compiled = torch.compile(lambda a, b: torch.mul(a, b))
compiled_result = compiled(x_device, y_device).cpu()

# Print the results and compare them
print(f"CPU result\n{cpu_result}")
print(f"Spyre Compiled result\n{compiled_result}")
cpu_delta = torch.abs(compiled_result - cpu_result).max()

print(f"Max delta Compiled Spyre vs. CPU: {cpu_delta}")
